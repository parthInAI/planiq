"""
PlanIQ — Test Suite: Step 4
Tests the FastAPI endpoints using TestClient.
Run: pytest tests/test_step4.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

from retrieval.hybrid_retriever import RetrievalResult, RetrievedChunk
from generation.engine import PlanIQResponse
from generation.prompts import MANDATORY_DISCLAIMER


# ── Mock factories ────────────────────────────────

def make_mock_chunk():
    return RetrievedChunk(
        text="Class 1 — Extension not exceeding 40 square metres is exempt.",
        metadata={
            "chunk_id": "test_001", "section_ref": "Class 1",
            "jurisdiction": "national", "source_title": "PDR 2001 Schedule 2",
            "confidence": "high", "is_stale": False,
            "effective_date": "2001-10-01", "needs_reverification": False,
        },
        rrf_score=0.02, rerank_score=5.0, source="both",
    )

def make_mock_retrieval(empty=False):
    chunks = [] if empty else [make_mock_chunk()]
    return RetrievalResult(
        query="test query", chunks=chunks, jurisdiction_used="national",
        total_dense_hits=5 if not empty else 0,
        total_sparse_hits=5 if not empty else 0,
        reranker_used=True, retrieval_quality=0.75 if not empty else 0.0,
    )

def make_mock_response(blocked=False, escalation=False):
    return PlanIQResponse(
        query="test query", query_type="EXEMPTION",
        answer_summary="Test answer summary.",
        full_answer={"is_exempt": True, "confidence": "high", "chunks_used": [1]},
        citations=[], confidence="blocked" if blocked else "high",
        confidence_score=0.0 if blocked else 0.8,
        disclaimer=MANDATORY_DISCLAIMER, user_warning="",
        requires_escalation=escalation, latency_ms=150,
        llm_provider="mock", is_blocked=blocked,
        block_reason="No chunks" if blocked else "",
    )

def make_mock_kb():
    mock_kb = MagicMock()
    mock_kb.get_stats.return_value = {
        "total_chunks_chroma": 1185, "total_chunks_bm25": 1185,
        "total_docs_ingested": 6, "last_updated": "2026-06-14T18:00:00",
        "stale_chunks_bm25": 0, "chunks_needing_reverify": 0,
    }
    mock_kb.list_sources.return_value = [
        {"doc_id": "doc1", "title": "PDA 2024", "type": "primary_act",
         "jurisdiction": "national", "chunk_count": 285}
    ]
    return mock_kb

# ── App fixture — patches lifespan so no real KB loads ───────────────────────

@pytest.fixture
def client():
    import api.main as main_module
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    mock_kb       = make_mock_kb()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = make_mock_retrieval()
    mock_engine   = MagicMock()
    mock_engine.generate.return_value = make_mock_response()
    mock_engine.provider_name = "mock"

    # Patch the lifespan so it doesn't try to load real KB
    @asynccontextmanager
    async def mock_lifespan(app):
        main_module._kb        = mock_kb
        main_module._retriever = mock_retriever
        main_module._engine    = mock_engine
        yield

    # Rebuild app with mock lifespan
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    test_app = FastAPI(lifespan=mock_lifespan)
    test_app.add_middleware(CORSMiddleware, allow_origins=["*"],
                            allow_methods=["*"], allow_headers=["*"])

    # Copy routes from main app
    for route in main_module.app.routes:
        test_app.routes.append(route)

    with TestClient(test_app) as c:
        yield c, mock_kb, mock_retriever, mock_engine


# ── Health ────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        c, *_ = client
        assert c.get("/health").status_code == 200

    def test_health_returns_healthy_status(self, client):
        c, *_ = client
        assert c.get("/health").json()["status"] == "healthy"

    def test_health_returns_chunk_count(self, client):
        c, *_ = client
        assert c.get("/health").json()["kb_chunks"] == 1185

    def test_health_has_timestamp(self, client):
        c, *_ = client
        assert "timestamp" in c.get("/health").json()

    def test_health_has_version(self, client):
        c, *_ = client
        assert "version" in c.get("/health").json()


# ── Stats ─────────────────────────────────────────

class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        c, *_ = client
        assert c.get("/stats").status_code == 200

    def test_stats_has_chunks(self, client):
        c, *_ = client
        assert c.get("/stats").json()["total_chunks"] == 1185

    def test_stats_has_sources(self, client):
        c, *_ = client
        data = c.get("/stats").json()
        assert "sources" in data and len(data["sources"]) > 0


# ── Councils ──────────────────────────────────────

class TestCouncilsEndpoint:
    def test_councils_returns_200(self, client):
        c, *_ = client
        assert c.get("/councils").status_code == 200

    def test_councils_returns_31(self, client):
        c, *_ = client
        assert c.get("/councils").json()["total"] == 31

    def test_dublin_city_in_councils(self, client):
        c, *_ = client
        slugs = [x["slug"] for x in c.get("/councils").json()["councils"]]
        assert "dublin_city" in slugs

    def test_fingal_in_councils(self, client):
        c, *_ = client
        slugs = [x["slug"] for x in c.get("/councils").json()["councils"]]
        assert "fingal" in slugs

    def test_councils_sorted(self, client):
        c, *_ = client
        names = [x["name"] for x in c.get("/councils").json()["councils"]]
        assert names == sorted(names)


# ── Query ─────────────────────────────────────────

class TestQueryEndpoint:
    def test_valid_query_200(self, client):
        c, *_ = client
        assert c.post("/query", json={"query": "Do I need planning permission for a rear extension?", "council": "dublin_city"}).status_code == 200

    def test_response_has_request_id(self, client):
        c, *_ = client
        data = c.post("/query", json={"query": "Is a garden shed exempt?", "council": "national"}).json()
        assert data["request_id"].startswith("planiq_")

    def test_response_has_answer(self, client):
        c, *_ = client
        data = c.post("/query", json={"query": "Is a garden shed exempt?", "council": "national"}).json()
        assert "summary" in data["answer"]

    def test_disclaimer_always_present(self, client):
        c, *_ = client
        data = c.post("/query", json={"query": "Is a garden shed exempt?", "council": "national"}).json()
        assert "professional planning advice" in data["answer"]["disclaimer"]

    def test_short_query_rejected(self, client):
        c, *_ = client
        assert c.post("/query", json={"query": "hi", "council": "national"}).status_code == 422

    def test_unknown_council_rejected(self, client):
        c, *_ = client
        assert c.post("/query", json={"query": "Do I need planning permission?", "council": "moon_base"}).status_code == 422

    def test_retriever_called(self, client):
        c, mk, mr, me = client
        c.post("/query", json={"query": "Is a conservatory exempt from planning?", "council": "national"})
        mr.retrieve.assert_called_once()

    def test_engine_called(self, client):
        c, mk, mr, me = client
        c.post("/query", json={"query": "Is a conservatory exempt from planning?", "council": "national"})
        me.generate.assert_called_once()

    def test_blocked_propagated(self, client):
        c, mk, mr, me = client
        me.generate.return_value = make_mock_response(blocked=True)
        data = c.post("/query", json={"query": "Is a conservatory exempt from planning?", "council": "national"}).json()
        assert data["answer"]["is_blocked"] is True

    def test_escalation_propagated(self, client):
        c, mk, mr, me = client
        me.generate.return_value = make_mock_response(escalation=True)
        data = c.post("/query", json={"query": "How do I appeal a planning refusal decision?", "council": "national"}).json()
        assert data["answer"]["escalation"] is True

    def test_meta_has_latency(self, client):
        c, *_ = client
        data = c.post("/query", json={"query": "Is a garden shed exempt from planning?", "council": "national"}).json()
        assert "latency_ms" in data["meta"]


# ── Feedback ──────────────────────────────────────

class TestFeedbackEndpoint:
    def test_positive_feedback(self, client):
        c, *_ = client
        resp = c.post("/feedback", json={"request_id": "planiq_abc", "query": "test question here", "helpful": True})
        assert resp.status_code == 200 and resp.json()["status"] == "received"

    def test_negative_feedback(self, client):
        c, *_ = client
        resp = c.post("/feedback", json={"request_id": "planiq_abc", "query": "test question here", "helpful": False, "comment": "Wrong threshold mentioned."})
        assert resp.status_code == 200

    def test_feedback_no_comment(self, client):
        c, *_ = client
        resp = c.post("/feedback", json={"request_id": "planiq_abc", "query": "test question here", "helpful": True})
        assert resp.status_code == 200
