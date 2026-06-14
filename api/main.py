"""
PlanIQ — FastAPI Application (Step 4)
=======================================
The REST API layer that connects the AI pipeline to any frontend.

Endpoints:
  POST /query          — main query endpoint (eligibility/exemption/process)
  GET  /health         — system health check
  GET  /stats          — knowledge base statistics
  GET  /councils       — list all 31 supported Irish councils
  POST /feedback       — user feedback on response quality

Design:
  - Single KB + retriever + engine instance (loaded once at startup)
  - Request validation via Pydantic models
  - Structured error responses — never raw Python exceptions to client
  - CORS enabled for local Streamlit development
  - Rate limiting headers returned (enforcement in Step 5)
  - Every response includes request_id for tracing
"""

import sys
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

from ingestion.schema import Jurisdiction
from knowledge_base.store import PlanIQKnowledgeBase
from retrieval.hybrid_retriever import HybridRetriever
from generation.engine import PlanIQGenerationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planiq.api")

# ── Global state (loaded once at startup) ─────────────────────────────────────
_kb:       Optional[PlanIQKnowledgeBase]    = None
_retriever: Optional[HybridRetriever]       = None
_engine:   Optional[PlanIQGenerationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the KB, retriever, and engine once at startup."""
    global _kb, _retriever, _engine

    logger.info("PlanIQ API starting up...")
    start = time.time()

    try:
        _kb       = PlanIQKnowledgeBase()
        _retriever = HybridRetriever(_kb)

        # Use mock provider by default — swap to "anthropic" in production
        import os
        provider = os.environ.get("PLANIQ_LLM_PROVIDER", "mock")
        _engine  = PlanIQGenerationEngine(provider=provider)

        elapsed = time.time() - start
        logger.info(f"PlanIQ ready in {elapsed:.1f}s | KB chunks: {_kb.get_stats()['total_chunks_chroma']}")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # API is running

    logger.info("PlanIQ API shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PlanIQ API",
    description = "AI-powered Irish planning permission guidance",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow Streamlit (localhost:8501) and future Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=10, max_length=500,
                            description="The planning question")
    council:    str = Field("national",
                            description="Irish council slug e.g. 'dublin_city', 'fingal'")
    top_k:      int = Field(5, ge=1, le=10,
                            description="Number of chunks to retrieve")

    model_config = {"json_schema_extra": {"example": {
        "query":  "Do I need planning permission to build a rear extension in Dublin?",
        "council": "dublin_city",
        "top_k":  5,
    }}}


class FeedbackRequest(BaseModel):
    request_id: str
    query:      str
    helpful:    bool
    comment:    Optional[str] = None


class HealthResponse(BaseModel):
    status:     str
    version:    str
    kb_chunks:  int
    timestamp:  str


class StatsResponse(BaseModel):
    total_chunks:     int
    total_docs:       int
    last_updated:     str
    chunks_by_source: dict


# ── Helper ────────────────────────────────────────────────────────────────────

def get_jurisdiction(council: str) -> Optional[Jurisdiction]:
    """Convert council slug to Jurisdiction enum. Returns None for 'national'."""
    if council == "national":
        return None
    try:
        return Jurisdiction(council)
    except ValueError:
        return None


def make_request_id() -> str:
    return f"planiq_{uuid.uuid4().hex[:12]}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — confirms API and KB are operational."""
    if _kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    stats = _kb.get_stats()
    return HealthResponse(
        status    = "healthy",
        version   = "1.0.0",
        kb_chunks = stats["total_chunks_chroma"],
        timestamp = datetime.utcnow().isoformat(),
    )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Knowledge base statistics — chunks, sources, last updated."""
    if _kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    stats   = _kb.get_stats()
    sources = _kb.list_sources()

    return {
        "total_chunks":     stats["total_chunks_chroma"],
        "total_docs":       stats["total_docs_ingested"],
        "last_updated":     stats["last_updated"],
        "bm25_chunks":      stats["total_chunks_bm25"],
        "stale_chunks":     stats["stale_chunks_bm25"],
        "needing_reverify": stats["chunks_needing_reverify"],
        "sources":          sources,
    }


@app.get("/councils", tags=["Reference"])
async def list_councils():
    """List all 31 supported Irish local authorities."""
    councils = [
        {"slug": j.value, "name": j.value.replace("_", " ").title()}
        for j in Jurisdiction
        if j != Jurisdiction.NATIONAL
    ]
    return {
        "total":   len(councils),
        "councils": sorted(councils, key=lambda c: c["name"]),
    }


@app.post("/query", tags=["Planning"])
async def query(request: QueryRequest, http_request: Request):
    """
    Main planning query endpoint.

    Runs the full pipeline:
      retrieve → hallucination check → generate → post-check → respond

    Returns a structured response with:
      - Plain English answer summary
      - Full structured answer (JSON)
      - Citations from source planning law
      - Confidence score and warnings
      - Mandatory disclaimer
    """
    request_id = make_request_id()
    start      = time.time()

    logger.info(f"[{request_id}] Query: {request.query[:60]}... | council={request.council}")

    # ── Validate council ──────────────────────────
    jurisdiction = get_jurisdiction(request.council)
    if request.council != "national" and jurisdiction is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown council: '{request.council}'. Use GET /councils for valid values."
        )

    # ── Retrieve ──────────────────────────────────
    try:
        retrieval = _retriever.retrieve(
            query        = request.query,
            jurisdiction = jurisdiction,
            top_k        = request.top_k,
            use_reranker = True,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    # ── Generate ──────────────────────────────────
    try:
        response = _engine.generate(
            query            = request.query,
            retrieval_result = retrieval,
            jurisdiction     = request.council,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Generation error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

    elapsed = int((time.time() - start) * 1000)
    logger.info(
        f"[{request_id}] Done | "
        f"confidence={response.confidence} | "
        f"blocked={response.is_blocked} | "
        f"latency={elapsed}ms"
    )

    # ── Return response ───────────────────────────
    return {
        "request_id":    request_id,
        "query":         request.query,
        "council":       request.council,
        "query_type":    response.query_type,
        "answer": {
            "summary":       response.answer_summary,
            "full":          response.full_answer,
            "citations":     response.citations,
            "confidence":    response.confidence,
            "warning":       response.user_warning,
            "disclaimer":    response.disclaimer,
            "escalation":    response.requires_escalation,
            "is_blocked":    response.is_blocked,
            "block_reason":  response.block_reason,
        },
        "meta": {
            "chunks_retrieved": len(retrieval.chunks),
            "retrieval_quality": retrieval.retrieval_quality,
            "latency_ms":    elapsed,
            "llm_provider":  response.llm_provider,
        },
    }


@app.post("/feedback", tags=["Quality"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Collect user feedback on response quality.
    Used to build the DPO training dataset for fine-tuning.
    Negative feedback (helpful=false) = candidate for DPO 'rejected' pairs.
    """
    logger.info(
        f"Feedback: request_id={feedback.request_id} | "
        f"helpful={feedback.helpful} | "
        f"comment={feedback.comment[:50] if feedback.comment else None}"
    )
    # In production: write to Supabase feedback table
    # For MVP: log only
    return {
        "status":  "received",
        "message": "Thank you — your feedback improves PlanIQ's accuracy.",
    }


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal server error",
            "message": "PlanIQ encountered an unexpected error. Please try again.",
        }
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,
        log_level = "info",
    )
