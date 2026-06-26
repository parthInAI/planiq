"""
PlanIQ — FastAPI Application v2.0
===================================
Endpoints:
  POST /query    — planning query (eligibility/exemption/process)
  POST /upload   — document review (Article 22 compliance check)
  GET  /health   — system health
  GET  /stats    — KB statistics
  GET  /councils — all 31 Irish councils
  POST /feedback — user feedback
"""

import sys
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.schema import Jurisdiction
from knowledge_base.store import PlanIQKnowledgeBase
from retrieval.hybrid_retriever import HybridRetriever
from generation.engine import PlanIQGenerationEngine
from document_review.pdf_extractor import PDFFieldExtractor
from document_review.article22_checker import Article22Checker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planiq.api")

_kb:        Optional[PlanIQKnowledgeBase]    = None
_retriever: Optional[HybridRetriever]        = None
_engine:    Optional[PlanIQGenerationEngine] = None
_extractor: Optional[PDFFieldExtractor]      = None
_checker:   Optional[Article22Checker]       = None

MAX_UPLOAD_MB    = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kb, _retriever, _engine, _extractor, _checker
    logger.info("PlanIQ API starting up...")
    start = time.time()
    try:
        _kb        = PlanIQKnowledgeBase()
        _retriever = HybridRetriever(_kb)
        import os
        provider = os.environ.get("PLANIQ_LLM_PROVIDER", "mock")
        _engine  = PlanIQGenerationEngine(provider=provider)
        _extractor = PDFFieldExtractor()
        _checker   = Article22Checker()
        elapsed = time.time() - start
        logger.info(f"PlanIQ ready in {elapsed:.1f}s | KB: {_kb.get_stats()['total_chunks_chroma']} chunks | Doc review: enabled")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("PlanIQ shutting down.")


app = FastAPI(title="PlanIQ API", description="AI-powered Irish planning guidance", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query:   str = Field(..., min_length=10, max_length=500)
    council: str = Field("national")
    top_k:   int = Field(5, ge=1, le=10)


class FeedbackRequest(BaseModel):
    request_id: str
    query:      str
    helpful:    bool
    comment:    Optional[str] = None


def get_jurisdiction(council: str) -> Optional[Jurisdiction]:
    if council == "national":
        return None
    try:
        return Jurisdiction(council)
    except ValueError:
        return None


def make_request_id() -> str:
    return f"planiq_{uuid.uuid4().hex[:12]}"


def serialise_report(report) -> dict:
    return {
        "application_description": report.application_description,
        "council":                 report.council,
        "checked_at":              report.checked_at,
        "total_checks":            report.total_checks,
        "passed":                  report.passed,
        "failed":                  report.failed,
        "warnings":                report.warnings,
        "missing":                 report.missing,
        "overall_status":          report.overall_status,
        "disclaimer":              report.disclaimer,
        "checks": [
            {"article": c.article, "item": c.item, "status": c.status.value,
             "finding": c.finding, "requirement": c.requirement,
             "action": c.action, "severity": c.severity}
            for c in report.checks
        ],
    }


@app.get("/health", tags=["System"])
async def health_check():
    if _kb is None:
        raise HTTPException(status_code=503, detail="KB not loaded")
    stats = _kb.get_stats()
    return {"status": "healthy", "version": "2.0.0",
            "kb_chunks": stats["total_chunks_chroma"],
            "document_review": _extractor is not None,
            "timestamp": datetime.utcnow().isoformat()}


@app.get("/stats", tags=["System"])
async def get_stats():
    if _kb is None:
        raise HTTPException(status_code=503, detail="KB not loaded")
    stats   = _kb.get_stats()
    sources = _kb.list_sources()
    return {"total_chunks": stats["total_chunks_chroma"],
            "total_docs": stats["total_docs_ingested"],
            "last_updated": stats["last_updated"],
            "bm25_chunks": stats["total_chunks_bm25"],
            "stale_chunks": stats["stale_chunks_bm25"],
            "needing_reverify": stats["chunks_needing_reverify"],
            "sources": sources}


@app.get("/councils", tags=["Reference"])
async def list_councils():
    councils = [{"slug": j.value, "name": j.value.replace("_", " ").title()}
                for j in Jurisdiction if j != Jurisdiction.NATIONAL]
    return {"total": len(councils), "councils": sorted(councils, key=lambda c: c["name"])}


@app.post("/query", tags=["Planning"])
async def query(request: QueryRequest, http_request: Request):
    request_id = make_request_id()
    start      = time.time()
    logger.info(f"[{request_id}] Query: {request.query[:60]} | council={request.council}")

    jurisdiction = get_jurisdiction(request.council)
    if request.council != "national" and jurisdiction is None:
        raise HTTPException(status_code=422, detail=f"Unknown council: '{request.council}'")

    try:
        retrieval = _retriever.retrieve(query=request.query, jurisdiction=jurisdiction,
                                        top_k=request.top_k, use_reranker=True)
    except Exception as e:
        logger.error(f"[{request_id}] Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    try:
        response = _engine.generate(query=request.query, retrieval_result=retrieval,
                                    jurisdiction=request.council)
    except Exception as e:
        logger.error(f"[{request_id}] Generation error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"[{request_id}] Done | confidence={response.confidence} | {elapsed}ms")

    return {
        "request_id": request_id, "query": request.query, "council": request.council,
        "query_type": response.query_type,
        "answer": {
            "summary": response.answer_summary, "full": response.full_answer,
            "citations": response.citations, "confidence": response.confidence,
            "warning": response.user_warning, "disclaimer": response.disclaimer,
            "escalation": response.requires_escalation, "is_blocked": response.is_blocked,
            "block_reason": response.block_reason,
        },
        "meta": {
            "chunks_retrieved": len(retrieval.chunks),
            "retrieval_quality": retrieval.retrieval_quality,
            "latency_ms": elapsed, "llm_provider": response.llm_provider,
        },
    }


@app.post("/upload", tags=["Document Review"])
async def upload_planning_application(
    file:    UploadFile = File(..., description="Planning application PDF — max 20MB"),
    council: str        = Form("national", description="Council slug e.g. dublin_city"),
):
    """
    Article 22 compliance check — upload a draft planning application PDF.

    PlanIQ extracts all fields and cross-checks them against Article 22 of the
    Planning and Development Regulations 2001. Returns a structured gap report
    showing PASS / FAIL / WARNING / MISSING for each requirement.

    This helps catch the issues that cause 35% of planning applications to be
    invalidated before the planning authority even assesses them.
    """
    request_id = make_request_id()
    start      = time.time()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=422, detail="No file provided.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail=f"Only PDF files accepted. Got: {file.filename}")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
            detail=f"File too large: {len(pdf_bytes)/1024/1024:.1f}MB. Max is {MAX_UPLOAD_MB}MB.")

    logger.info(f"[{request_id}] Upload: {file.filename} | {len(pdf_bytes)//1024}KB | council={council}")

    # ── Extract ───────────────────────────────────────────────────────────────
    try:
        fields = _extractor.extract(pdf_bytes)
    except Exception as e:
        logger.error(f"[{request_id}] Extraction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")

    if fields.get("raw_text_length", 0) < 100:
        return {
            "request_id": request_id, "filename": file.filename,
            "extraction_warning": (
                "Very little text was extracted. This may be a scanned image PDF. "
                "The gap report below may be incomplete. Consider using a text-based PDF."
            ),
            "gap_report": None,
            "meta": {"file_size_kb": len(pdf_bytes)//1024, "extraction_confidence": "low",
                     "latency_ms": int((time.time()-start)*1000)},
        }

    if council and council != "national" and not fields.get("planning_authority"):
        fields["planning_authority"] = council.replace("_", " ").title()

    # ── Check ─────────────────────────────────────────────────────────────────
    try:
        report = _checker.check(fields)
    except Exception as e:
        logger.error(f"[{request_id}] Checker error: {e}")
        raise HTTPException(status_code=500, detail="Article 22 compliance check failed.")

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"[{request_id}] Gap report: passed={report.passed} failed={report.failed} "
                f"warnings={report.warnings} status={report.overall_status} | {elapsed}ms")

    return {
        "request_id": request_id,
        "filename":   file.filename,
        "council":    council,
        "extracted_fields": {
            "applicant_name":          fields.get("applicant_name", ""),
            "planning_authority":      fields.get("planning_authority", ""),
            "development_description": fields.get("development_description", ""),
            "permission_type":         fields.get("permission_type", ""),
            "is_protected_structure":  fields.get("is_protected_structure", False),
            "is_aca":                  fields.get("is_aca", False),
            "extraction_confidence":   fields.get("extraction_confidence", ""),
        },
        "gap_report": serialise_report(report),
        "meta": {
            "file_size_kb":          len(pdf_bytes) // 1024,
            "text_chars_extracted":  fields.get("raw_text_length", 0),
            "extraction_confidence": fields.get("extraction_confidence", ""),
            "latency_ms":            elapsed,
        },
    }


@app.post("/feedback", tags=["Quality"])
async def submit_feedback(feedback: FeedbackRequest):
    logger.info(f"Feedback: {feedback.request_id} | helpful={feedback.helpful}")
    return {"status": "received", "message": "Thank you — your feedback improves PlanIQ."}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500,
        content={"error": "Internal server error", "message": "Please try again."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
