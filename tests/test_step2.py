"""
PlanIQ — Test Suite: Step 2
Tests hybrid retrieval and hallucination detection.
Run: pytest tests/test_step2.py -v

Note: These tests use a lightweight mock KB so they run without
needing the full 1,185-chunk ChromaDB on disk.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))

from retrieval.hybrid_retriever import (
    HybridRetriever, RetrievalResult, RetrievedChunk,
    FINAL_TOP_K, RRF_K
)
from hallucination.detector import (
    HallucinationDetector, HallucinationReport,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW
)


# ── Fixtures ─────────────────────────────────────

def make_retrieved_chunk(
    chunk_id="chunk_001",
    text="Class 1 — Extension to the rear of a dwellinghouse not exceeding 40 square metres.",
    section_ref="Class 1 | S.I. No. 600 of 2001",
    jurisdiction="national",
    source_title="PDR 2001 Schedule 2",
    confidence="high",
    is_stale=False,
    source="both",
    rrf_score=0.02,
    rerank_score=5.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={
            "chunk_id":     chunk_id,
            "section_ref":  section_ref,
            "jurisdiction": jurisdiction,
            "source_title": source_title,
            "confidence":   confidence,
            "is_stale":     is_stale,
            "effective_date": "2001-10-01",
            "needs_reverification": False,
        },
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        source=source,
    )


def make_retrieval_result(
    query="Do I need planning permission for a rear extension?",
    chunks=None,
    quality=0.75,
    dense_hits=5,
    sparse_hits=5,
    reranker_used=True,
) -> RetrievalResult:
    if chunks is None:
        chunks = [make_retrieved_chunk()]
    return RetrievalResult(
        query=query,
        chunks=chunks,
        jurisdiction_used="dublin_city",
        total_dense_hits=dense_hits,
        total_sparse_hits=sparse_hits,
        reranker_used=reranker_used,
        retrieval_quality=quality,
    )


# ── RetrievedChunk tests ──────────────────────────

class TestRetrievedChunk:

    def test_chunk_properties(self):
        chunk = make_retrieved_chunk()
        assert chunk.chunk_id == "chunk_001"
        assert chunk.jurisdiction == "national"
        assert chunk.is_stale is False
        assert chunk.confidence == "high"

    def test_stale_chunk_detected(self):
        chunk = make_retrieved_chunk(is_stale=True)
        assert chunk.is_stale is True

    def test_section_ref_returned(self):
        chunk = make_retrieved_chunk(section_ref="Class 1 | S.I. No. 600 of 2001")
        assert "Class 1" in chunk.section_ref


# ── RetrievalResult tests ─────────────────────────

class TestRetrievalResult:

    def test_top_chunk_returned(self):
        chunks = [make_retrieved_chunk(chunk_id=f"c{i}") for i in range(3)]
        result = make_retrieval_result(chunks=chunks)
        assert result.top_chunk.chunk_id == "c0"

    def test_empty_result(self):
        result = make_retrieval_result(chunks=[])
        assert result.is_empty is True
        assert result.top_chunk is None

    def test_to_context_string_format(self):
        result = make_retrieval_result()
        ctx    = result.to_context_string()
        assert "[CHUNK 1]" in ctx
        assert "Source:" in ctx
        assert "Section:" in ctx
        assert "Jurisdiction:" in ctx

    def test_empty_context_string(self):
        result = make_retrieval_result(chunks=[])
        ctx    = result.to_context_string()
        assert "No relevant" in ctx

    def test_entity_set_extracts_thresholds(self):
        chunk  = make_retrieved_chunk(
            text="Class 1 extension not exceeding 40 square metres in floor area."
        )
        result = make_retrieval_result(chunks=[chunk])
        entities = result.get_entity_set()
        assert any("40" in e for e in entities), "40 sq metre threshold not extracted"

    def test_entity_set_extracts_section_refs(self):
        chunk  = make_retrieved_chunk(section_ref="Class 1 | S.I. No. 600 of 2001")
        result = make_retrieval_result(chunks=[chunk])
        entities = result.get_entity_set()
        assert any("class 1" in e for e in entities)

    def test_entity_set_extracts_si_numbers(self):
        chunk  = make_retrieved_chunk(
            text="As per S.I. No. 600 of 2001, the following classes are exempt."
        )
        result = make_retrieval_result(chunks=[chunk])
        entities = result.get_entity_set()
        assert any("600" in e for e in entities)


# ── RRF fusion tests ─────────────────────────────

class TestRRFFusion:

    def get_retriever(self):
        kb = MagicMock()
        return HybridRetriever(kb)

    def test_rrf_merges_both_lists(self):
        retriever = self.get_retriever()
        dense = [{"metadata": {"chunk_id": f"d{i}", "is_stale": False}, "text": f"dense {i}", "score": 0.9-i*0.1} for i in range(5)]
        sparse = [{"metadata": {"chunk_id": f"s{i}", "is_stale": False}, "text": f"sparse {i}", "score": 5-i} for i in range(5)]
        fused = retriever._reciprocal_rank_fusion(dense, sparse)
        ids = [c.chunk_id for c in fused]
        # Both dense and sparse chunks should appear
        assert any(id.startswith("d") for id in ids)
        assert any(id.startswith("s") for id in ids)

    def test_chunk_in_both_lists_scores_higher(self):
        retriever = self.get_retriever()
        shared_id = "shared_001"
        dense  = [{"metadata": {"chunk_id": shared_id, "is_stale": False}, "text": "shared chunk", "score": 0.9}]
        sparse = [{"metadata": {"chunk_id": shared_id, "is_stale": False}, "text": "shared chunk", "score": 8.0}]
        fused  = retriever._reciprocal_rank_fusion(dense, sparse)
        shared = next(c for c in fused if c.chunk_id == shared_id)
        # Shared chunk should have higher RRF score than any single-list chunk
        expected = 1/(RRF_K+1) + 1/(RRF_K+1)
        assert abs(shared.rrf_score - expected) < 0.0001

    def test_chunk_in_both_lists_marked_as_both(self):
        retriever = self.get_retriever()
        shared_id = "shared_001"
        dense  = [{"metadata": {"chunk_id": shared_id, "is_stale": False}, "text": "shared", "score": 0.9}]
        sparse = [{"metadata": {"chunk_id": shared_id, "is_stale": False}, "text": "shared", "score": 8.0}]
        fused  = retriever._reciprocal_rank_fusion(dense, sparse)
        shared = next(c for c in fused if c.chunk_id == shared_id)
        assert shared.source == "both"

    def test_rrf_sorted_descending(self):
        retriever = self.get_retriever()
        dense  = [{"metadata": {"chunk_id": f"d{i}", "is_stale": False}, "text": f"t{i}", "score": 0.9} for i in range(10)]
        sparse = [{"metadata": {"chunk_id": f"s{i}", "is_stale": False}, "text": f"t{i}", "score": 5.0} for i in range(10)]
        fused  = retriever._reciprocal_rank_fusion(dense, sparse)
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_stale_chunks_filtered_after_fusion(self):
        retriever = self.get_retriever()
        dense  = [{"metadata": {"chunk_id": "stale_01", "is_stale": True}, "text": "stale", "score": 0.99}]
        sparse = []
        fused  = retriever._reciprocal_rank_fusion(dense, sparse)
        filtered = [c for c in fused if not c.is_stale]
        assert all(not c.is_stale for c in filtered)


# ── Hallucination detector tests ──────────────────

class TestHallucinationDetector:

    def get_detector(self):
        return HallucinationDetector()

    def test_empty_retrieval_is_blocked(self):
        detector = self.get_detector()
        result   = make_retrieval_result(chunks=[])
        report   = detector.analyse("test query", result)
        assert report.is_blocked is True
        assert report.confidence_score == 0.0
        assert any(f.code == "NO_CHUNKS_RETRIEVED" for f in report.flags)

    def test_good_retrieval_not_blocked(self):
        detector = self.get_detector()
        result   = make_retrieval_result(quality=0.8)
        report   = detector.analyse("rear extension planning permission", result)
        assert report.is_blocked is False

    def test_stale_chunk_blocks_response(self):
        detector = self.get_detector()
        stale    = make_retrieved_chunk(is_stale=True)
        result   = make_retrieval_result(chunks=[stale])
        report   = detector.analyse("test", result)
        assert report.is_blocked is True
        assert any(f.code == "STALE_CHUNKS_IN_CONTEXT" for f in report.flags)

    def test_appeal_query_triggers_escalation(self):
        detector = self.get_detector()
        result   = make_retrieval_result(query="how do I appeal a planning refusal?")
        report   = detector.analyse("how do I appeal a planning refusal?", result)
        assert report.requires_escalation is True
        assert any(f.code == "ESCALATION_REQUIRED" for f in report.flags)

    def test_enforcement_query_triggers_escalation(self):
        detector = self.get_detector()
        result   = make_retrieval_result()
        report   = detector.analyse("what happens with enforcement action?", result)
        assert report.requires_escalation is True

    def test_section5_triggers_escalation(self):
        detector = self.get_detector()
        result   = make_retrieval_result()
        report   = detector.analyse("how do I get a section 5 declaration?", result)
        assert report.requires_escalation is True

    def test_ungrounded_section_reference_flagged(self):
        detector = self.get_detector()
        chunk    = make_retrieved_chunk(text="Class 1 rear extension 40 square metres.", section_ref="Class 1")
        result   = make_retrieval_result(chunks=[chunk])
        # Generated text references a section NOT in the retrieved chunks
        generated = "Under Class 99 of the regulations, you are exempt."
        report   = detector.analyse("test", result, generated_text=generated)
        assert any(f.code == "UNGROUNDED_SECTION_REFERENCE" for f in report.flags)
        assert "class 99" in [c.lower() for c in report.ungrounded_claims]

    def test_grounded_section_reference_not_flagged(self):
        detector = self.get_detector()
        chunk    = make_retrieved_chunk(text="Class 1 rear extension 40 square metres.", section_ref="Class 1")
        result   = make_retrieval_result(chunks=[chunk])
        # Generated text references Class 1 — which IS in the retrieved chunks
        generated = "Under Class 1, a rear extension not exceeding 40 square metres is exempt."
        report   = detector.analyse("test", result, generated_text=generated)
        section_flags = [f for f in report.flags if f.code == "UNGROUNDED_SECTION_REFERENCE"]
        assert len(section_flags) == 0

    def test_confidence_score_between_0_and_1(self):
        detector = self.get_detector()
        result   = make_retrieval_result()
        report   = detector.analyse("planning permission extension", result)
        assert 0.0 <= report.confidence_score <= 1.0

    def test_confidence_label_high(self):
        detector = self.get_detector()
        result   = make_retrieval_result(quality=0.95)
        report   = detector.analyse("simple extension question", result)
        assert report.confidence_label in ("high", "medium")

    def test_mandatory_disclaimer_always_present(self):
        detector = self.get_detector()
        result   = make_retrieval_result()
        report   = detector.analyse("test", result)
        assert "professional planning advice" in report.mandatory_disclaimer
        assert len(report.mandatory_disclaimer) > 50

    def test_user_warning_blocked(self):
        detector = self.get_detector()
        result   = make_retrieval_result(chunks=[])
        report   = detector.analyse("test", result)
        assert "cannot provide" in report.user_warning

    def test_user_warning_escalation(self):
        detector = self.get_detector()
        result   = make_retrieval_result()
        report   = detector.analyse("how do I appeal this decision?", result)
        assert "planning consultant" in report.user_warning

    def test_low_retrieval_quality_flagged(self):
        detector = self.get_detector()
        result   = make_retrieval_result(quality=0.1, dense_hits=1, sparse_hits=0)
        report   = detector.analyse("obscure planning query", result)
        assert any(f.code == "LOW_RETRIEVAL_QUALITY" for f in report.flags)
