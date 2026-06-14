"""
PlanIQ — Test Suite: Step 3
Tests the generation engine, prompt builder, and response assembly.
Run: pytest tests/test_step3.py -v

All tests use MockProvider — no API key or internet required.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

from retrieval.hybrid_retriever import RetrievalResult, RetrievedChunk
from generation.engine import PlanIQGenerationEngine, PlanIQResponse
from generation.prompts import (
    SYSTEM_PROMPT, ELIGIBILITY_PROMPT, EXEMPTION_PROMPT,
    PROCESS_PROMPT, MANDATORY_DISCLAIMER
)


# ── Fixtures ──────────────────────────────────────

def make_chunk(chunk_id="c1", text="Class 1 extension up to 40 square metres is exempt.", section_ref="Class 1", is_stale=False):
    return RetrievedChunk(
        text=text,
        metadata={
            "chunk_id":     chunk_id,
            "section_ref":  section_ref,
            "jurisdiction": "national",
            "source_title": "PDR 2001 Schedule 2",
            "confidence":   "high",
            "is_stale":     is_stale,
            "effective_date": "2001-10-01",
            "needs_reverification": False,
        },
        rrf_score=0.02,
        rerank_score=5.0,
        source="both",
    )


def make_retrieval(query="Do I need planning permission?", chunks=None, quality=0.75):
    if chunks is None:
        chunks = [make_chunk()]
    return RetrievalResult(
        query=query,
        chunks=chunks,
        jurisdiction_used="dublin_city",
        total_dense_hits=5,
        total_sparse_hits=5,
        reranker_used=True,
        retrieval_quality=quality,
    )


def get_engine():
    return PlanIQGenerationEngine(provider="mock")


# ── Prompt tests ──────────────────────────────────

class TestPrompts:

    def test_system_prompt_contains_citation_rule(self):
        assert "CITE BEFORE YOU CLAIM" in SYSTEM_PROMPT

    def test_system_prompt_contains_no_fabrication_rule(self):
        assert "NEVER INVENT SECTION NUMBERS" in SYSTEM_PROMPT

    def test_system_prompt_contains_threshold_rule(self):
        assert "NEVER INVENT THRESHOLDS" in SYSTEM_PROMPT

    def test_system_prompt_contains_irish_terminology(self):
        assert "An Coimisi" in SYSTEM_PROMPT

    def test_eligibility_prompt_has_json_schema(self):
        assert "permission_required" in ELIGIBILITY_PROMPT
        assert "reasoning" in ELIGIBILITY_PROMPT
        assert "citation" in ELIGIBILITY_PROMPT

    def test_exemption_prompt_has_thresholds_field(self):
        assert "thresholds" in EXEMPTION_PROMPT
        assert "is_exempt" in EXEMPTION_PROMPT
        assert "section_5_recommended" in EXEMPTION_PROMPT

    def test_process_prompt_has_deadlines_field(self):
        assert "key_deadlines" in PROCESS_PROMPT
        assert "consequence" in PROCESS_PROMPT

    def test_mandatory_disclaimer_present(self):
        assert "professional planning advice" in MANDATORY_DISCLAIMER
        assert "Section 5 declaration" in MANDATORY_DISCLAIMER

    def test_eligibility_prompt_formats_correctly(self):
        filled = ELIGIBILITY_PROMPT.format(
            context="Test context",
            query="Do I need permission?",
            jurisdiction="dublin_city",
        )
        assert "Test context" in filled
        assert "dublin_city" in filled


# ── Query classifier tests ────────────────────────

class TestQueryClassifier:

    def get_engine(self):
        return PlanIQGenerationEngine(provider="mock")

    def test_extension_classified_as_exemption(self):
        engine = self.get_engine()
        result = engine._classify_query("Do I need permission to extend my house?")
        assert result in ("EXEMPTION", "ELIGIBILITY")

    def test_shed_classified_as_exemption(self):
        engine = self.get_engine()
        result = engine._classify_query("Is a garden shed exempt from planning?")
        assert result == "EXEMPTION"

    def test_solar_classified_as_exemption(self):
        engine = self.get_engine()
        result = engine._classify_query("Are solar panels exempt from planning permission?")
        assert result == "EXEMPTION"

    def test_appeal_classified_as_process(self):
        engine = self.get_engine()
        result = engine._classify_query("How do I appeal a planning decision?")
        assert result == "PROCESS"

    def test_apply_classified_as_process(self):
        engine = self.get_engine()
        result = engine._classify_query("How do I apply for planning permission?")
        assert result == "PROCESS"

    def test_unknown_defaults_to_eligibility(self):
        engine = self.get_engine()
        result = engine._classify_query("What is a development plan?")
        assert result == "ELIGIBILITY"


# ── JSON parsing tests ────────────────────────────

class TestJSONParsing:

    def test_valid_json_parsed_correctly(self):
        engine = get_engine()
        raw    = json.dumps({"answer_summary": "Test", "confidence": "high", "chunks_used": [1]})
        parsed = engine._parse_response(raw, "ELIGIBILITY")
        assert parsed["answer_summary"] == "Test"
        assert parsed["confidence"] == "high"

    def test_json_with_markdown_fences_parsed(self):
        engine = get_engine()
        raw    = '```json\n{"answer_summary": "Test", "confidence": "medium", "chunks_used": []}\n```'
        parsed = engine._parse_response(raw, "ELIGIBILITY")
        assert parsed["answer_summary"] == "Test"

    def test_invalid_json_returns_safe_default(self):
        engine = get_engine()
        parsed = engine._parse_response("this is not json at all", "ELIGIBILITY")
        assert "answer_summary" in parsed
        assert "_parse_error" in parsed

    def test_empty_string_returns_safe_default(self):
        engine = get_engine()
        parsed = engine._parse_response("", "ELIGIBILITY")
        assert "answer_summary" in parsed


# ── Confidence logic tests ────────────────────────

class TestConfidenceLogic:

    def test_high_beats_medium(self):
        engine = get_engine()
        result = engine._lowest_confidence("high", "medium")
        assert result == "medium"

    def test_blocked_always_wins(self):
        engine = get_engine()
        result = engine._lowest_confidence("high", "blocked")
        assert result == "blocked"

    def test_same_level_returns_same(self):
        engine = get_engine()
        result = engine._lowest_confidence("medium", "medium")
        assert result == "medium"

    def test_low_beats_high(self):
        engine = get_engine()
        result = engine._lowest_confidence("high", "low")
        assert result == "low"


# ── Full generation pipeline tests ────────────────

class TestGenerationPipeline:

    def test_generate_returns_response_object(self):
        engine   = get_engine()
        retrieval = make_retrieval("Do I need planning permission for a rear extension?")
        response = engine.generate(
            query=retrieval.query,
            retrieval_result=retrieval,
            jurisdiction="dublin_city",
        )
        assert isinstance(response, PlanIQResponse)

    def test_response_has_answer_summary(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "dublin_city")
        assert response.answer_summary
        assert len(response.answer_summary) > 10

    def test_response_has_disclaimer(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "national")
        assert "professional planning advice" in response.disclaimer

    def test_empty_retrieval_returns_blocked(self):
        engine   = get_engine()
        retrieval = make_retrieval(chunks=[])
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.is_blocked is True
        assert response.confidence == "blocked"

    def test_blocked_response_has_no_full_answer(self):
        engine   = get_engine()
        retrieval = make_retrieval(chunks=[])
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.full_answer == {}

    def test_appeal_query_requires_escalation(self):
        engine   = get_engine()
        retrieval = make_retrieval(query="How do I appeal this planning refusal?")
        response = engine.generate(retrieval.query, retrieval, "dublin_city")
        assert response.requires_escalation is True

    def test_response_has_jurisdiction(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "fingal")
        assert response.jurisdiction == "fingal"

    def test_latency_recorded(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.latency_ms >= 0

    def test_provider_name_recorded(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.llm_provider == "mock"

    def test_to_display_dict_has_required_keys(self):
        engine   = get_engine()
        retrieval = make_retrieval()
        response = engine.generate(retrieval.query, retrieval, "national")
        display  = response.to_display_dict()
        required = ["query", "query_type", "summary", "answer",
                    "citations", "confidence", "warning", "disclaimer",
                    "escalation", "is_blocked"]
        for key in required:
            assert key in display, f"Missing key in display dict: {key}"

    def test_stale_chunk_produces_blocked_response(self):
        engine   = get_engine()
        stale    = make_chunk(is_stale=True)
        retrieval = make_retrieval(chunks=[stale])
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.is_blocked is True

    def test_exemption_query_classified_correctly(self):
        engine   = get_engine()
        retrieval = make_retrieval(query="Is a garden shed exempt from planning?")
        response = engine.generate(retrieval.query, retrieval, "dublin_city")
        assert response.query_type == "EXEMPTION"

    def test_process_query_classified_correctly(self):
        engine   = get_engine()
        retrieval = make_retrieval(query="How do I apply for planning permission?")
        response = engine.generate(retrieval.query, retrieval, "national")
        assert response.query_type == "PROCESS"
