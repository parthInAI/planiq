"""
PlanIQ — Generation Engine (Step 3)
=====================================
Takes a RetrievalResult + HallucinationReport and produces a
structured, cited, grounded answer to the user's planning query.

Pipeline:
  1. Query classification  (ELIGIBILITY / EXEMPTION / PROCESS)
  2. Prompt construction   (context + query + jurisdiction)
  3. LLM call              (temperature=0, structured JSON output)
  4. Post-generation check (entity grounding on actual output)
  5. Response assembly     (answer + citations + disclaimer + warnings)

LLM support:
  - Anthropic Claude API  (production — best reasoning)
  - Ollama local models   (development — free, private)
  - Mock mode             (testing — no API key needed)
"""

import os
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hallucination"))

from retrieval.hybrid_retriever import RetrievalResult
from hallucination.detector import HallucinationDetector, HallucinationReport
from generation.prompts import (
    SYSTEM_PROMPT, ELIGIBILITY_PROMPT, EXEMPTION_PROMPT,
    PROCESS_PROMPT, QUERY_CLASSIFIER_PROMPT, MANDATORY_DISCLAIMER
)

console = Console()

QueryType = Literal["ELIGIBILITY", "EXEMPTION", "PROCESS", "OTHER"]


# ── Response schema ───────────────────────────────────────────────────────────

@dataclass
class PlanIQResponse:
    """
    The complete structured response returned to the user interface.
    Every field is intentional — nothing is free text only.
    """
    # Core answer
    query:           str
    query_type:      QueryType
    answer_summary:  str          # one sentence, plain English
    full_answer:     dict         # parsed JSON from LLM

    # Provenance — shown to user
    citations:       list[dict] = field(default_factory=list)
    chunks_used:     list[int]  = field(default_factory=list)
    jurisdiction:    str        = "national"

    # Trust signals — shown to user
    confidence:      str        = "medium"   # high / medium / low / blocked
    confidence_score: float     = 0.0
    disclaimer:      str        = MANDATORY_DISCLAIMER
    user_warning:    str        = ""
    requires_escalation: bool   = False

    # Operational
    latency_ms:      int        = 0
    llm_provider:    str        = ""
    is_blocked:      bool       = False
    block_reason:    str        = ""

    def to_display_dict(self) -> dict:
        """Serialise for UI rendering."""
        return {
            "query":         self.query,
            "query_type":    self.query_type,
            "summary":       self.answer_summary,
            "answer":        self.full_answer,
            "citations":     self.citations,
            "confidence":    self.confidence,
            "warning":       self.user_warning,
            "disclaimer":    self.disclaimer,
            "escalation":    self.requires_escalation,
            "is_blocked":    self.is_blocked,
            "block_reason":  self.block_reason,
        }


# ── LLM providers ─────────────────────────────────────────────────────────────

class AnthropicProvider:
    """Calls Claude via Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            console.log(f"[green]✓[/] Anthropic provider ready: {model}")
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        message = self.client.messages.create(
            model       = self.model,
            max_tokens  = 1500,
            temperature = temperature,
            system      = system,
            messages    = [{"role": "user", "content": user}],
        )
        return message.content[0].text


class OllamaProvider:
    """Calls a local Ollama model — free, private, no API key."""

    def __init__(self, model: str = "mistral"):
        self.model   = model
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        console.log(f"[green]✓[/] Ollama provider ready: {model} @ {self.base_url}")

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        import requests
        payload = {
            "model":  self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class MockProvider:
    """
    Mock LLM for testing — returns structured JSON without any API call.
    Always used in tests. Zero cost, zero latency.
    """

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        # Detect query type from prompt
        if "ELIGIBILITY_PROMPT" in user or "permission_required" in user:
            return json.dumps({
                "answer_summary": "Based on the retrieved planning law, planning permission may be required for this development.",
                "permission_required": "uncertain",
                "reasoning": [{"point": "The works may constitute development under the Planning and Development Act 2000.", "citation": "[CHUNK 1 — PDA 2000, Section 3]"}],
                "conditions": ["Subject to local development plan zoning"],
                "what_to_do_next": "Submit a Section 5 declaration to your local planning authority to obtain a formal determination.",
                "confidence": "medium",
                "chunks_used": [1, 2],
            })
        elif "is_exempt" in user:
            return json.dumps({
                "answer_summary": "The described works appear to qualify as exempted development under Class 1 of Schedule 2.",
                "is_exempt": True,
                "exemption_class": "Class 1, Schedule 2, PDR 2001",
                "conditions": [{"condition": "Floor area must not exceed 40 square metres", "citation": "[CHUNK 1 — PDR 2001 Schedule 2, Class 1]"}],
                "thresholds": [{"threshold": "40 square metres maximum floor area", "citation": "[CHUNK 1 — PDR 2001 Schedule 2, Class 1]"}],
                "disqualifiers": ["Previous extensions count toward the 40 sq m limit"],
                "section_5_recommended": True,
                "what_to_do_next": "Obtain a Section 5 declaration from Dublin City Council to formally confirm the exemption.",
                "confidence": "high",
                "chunks_used": [1],
            })
        else:
            return json.dumps({
                "answer_summary": "The planning application process in Ireland involves several key stages.",
                "process_steps": [{"step": 1, "action": "Submit application to local authority", "timeline": "Week 0", "citation": "[CHUNK 1 — PDA 2000]"}],
                "key_deadlines": [{"deadline": "Third party appeal window", "timeframe": "4 weeks from decision", "consequence": "Loss of appeal rights if missed"}],
                "fees": None,
                "what_to_do_next": "Contact your local planning authority to begin the pre-application consultation process.",
                "confidence": "medium",
                "chunks_used": [1, 2, 3],
            })


# ── Generation engine ─────────────────────────────────────────────────────────

class PlanIQGenerationEngine:
    """
    Orchestrates the full generation pipeline for one user query.

    Usage:
        engine   = PlanIQGenerationEngine(provider="mock")
        response = engine.generate(query, retrieval_result, jurisdiction)
    """

    def __init__(
        self,
        provider: str = "mock",
        model:    str = "",
    ):
        self.detector = HallucinationDetector()
        self.llm      = self._init_provider(provider, model)
        self.provider_name = provider
        console.log(f"[dim]Generation engine ready | provider={provider}[/]")

    def generate(
        self,
        query:            str,
        retrieval_result: RetrievalResult,
        jurisdiction:     str = "national",
    ) -> PlanIQResponse:
        """
        Full generation pipeline — retrieval result → structured response.
        """
        start = time.time()

        # ── Pre-generation hallucination check ────
        pre_report = self.detector.analyse(query, retrieval_result)

        # If blocked before generation — return block response immediately
        if pre_report.is_blocked:
            return self._blocked_response(query, pre_report, start)

        # ── Classify query type ───────────────────
        query_type = self._classify_query(query)
        console.log(f"[dim]Query type: {query_type}[/]")

        # ── Build prompt ──────────────────────────
        context = retrieval_result.to_context_string()
        prompt  = self._build_prompt(query_type, context, query, jurisdiction)

        # ── LLM call (temperature=0) ──────────────
        try:
            raw_response = self.llm.generate(
                system      = SYSTEM_PROMPT,
                user        = prompt,
                temperature = 0.0,
            )
        except Exception as e:
            console.log(f"[red]LLM error: {e}[/]")
            return self._error_response(query, str(e), start)

        # ── Parse structured JSON output ──────────
        parsed = self._parse_response(raw_response, query_type)

        # ── Post-generation hallucination check ───
        post_report = self.detector.analyse(
            query, retrieval_result, generated_text=raw_response
        )

        # ── Assemble final response ───────────────
        response = self._assemble_response(
            query         = query,
            query_type    = query_type,
            parsed        = parsed,
            retrieval     = retrieval_result,
            report        = post_report,
            jurisdiction  = jurisdiction,
            start         = start,
        )

        self._log_response(response)
        return response

    # ── Private: provider init ────────────────────

    def _init_provider(self, provider: str, model: str):
        if provider == "anthropic":
            return AnthropicProvider(model or "claude-sonnet-4-6")
        elif provider == "ollama":
            return OllamaProvider(model or "mistral")
        else:
            return MockProvider()

    # ── Private: query classification ─────────────

    def _classify_query(self, query: str) -> QueryType:
        """
        Rule-based classification — fast, no LLM call needed.
        Falls back to keyword matching.
        """
        q = query.lower()

        exemption_keywords = [
            "exempt", "exempted", "exemption", "permission needed",
            "do i need", "require permission", "need planning",
            "without permission", "shed", "solar", "fence", "gate",
            "attic", "garage conversion", "porch",
        ]
        eligibility_keywords = [
            "need planning", "require planning", "planning permission for",
            "do i need permission", "is permission required",
            "extension", "conservatory", "outbuilding",
        ]
        process_keywords = [
            "how do i apply", "how to apply", "application process",
            "how long does", "appeal", "timeline", "what documents",
            "how much does", "planning fee", "submit", "observation",
        ]

        # Check exemption first (most specific)
        if any(k in q for k in exemption_keywords):
            return "EXEMPTION"
        if any(k in q for k in eligibility_keywords):
            return "ELIGIBILITY"
        if any(k in q for k in process_keywords):
            return "PROCESS"

        return "ELIGIBILITY"  # safe default

    # ── Private: prompt building ──────────────────

    def _build_prompt(
        self,
        query_type:   QueryType,
        context:      str,
        query:        str,
        jurisdiction: str,
    ) -> str:
        template_map = {
            "ELIGIBILITY": ELIGIBILITY_PROMPT,
            "EXEMPTION":   EXEMPTION_PROMPT,
            "PROCESS":     PROCESS_PROMPT,
            "OTHER":       ELIGIBILITY_PROMPT,  # fallback
        }
        template = template_map.get(query_type, ELIGIBILITY_PROMPT)
        return template.format(
            context=context,
            query=query,
            jurisdiction=jurisdiction,
        )

    # ── Private: JSON parsing ─────────────────────

    def _parse_response(self, raw: str, query_type: QueryType) -> dict:
        """
        Parse LLM JSON output robustly.
        Tries multiple strategies to extract valid JSON.
        """
        import re as _re
        clean = raw.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown code fences
        if "```" in clean:
            fence_match = _re.search(r'```(?:json)?\s*([\s\S]*?)```', clean)
            if fence_match:
                try:
                    return json.loads(fence_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Extract first {...} block from anywhere in text
        brace_match = _re.search(r'\{[\s\S]*\}', clean)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Safe fallback — show raw text as the answer
        console.log(f"[yellow]⚠ All JSON strategies failed[/]")
        summary = _re.sub(r'[{}\[\]]', '', raw[:300]).strip()
        return {
            "answer_summary": summary or "Please try rephrasing your question.",
            "confidence": "low",
            "chunks_used": [1],
            "reasoning": [{"point": raw[:400], "citation": "See retrieved chunks"}],
            "what_to_do_next": "Consult your local planning authority for a formal determination.",
        }

    # ── Private: response assembly ────────────────

    def _assemble_response(
        self,
        query:        str,
        query_type:   QueryType,
        parsed:       dict,
        retrieval:    RetrievalResult,
        report:       HallucinationReport,
        jurisdiction: str,
        start:        float,
    ) -> PlanIQResponse:
        """Assemble all signals into the final PlanIQResponse."""

        # Extract citations from parsed answer
        citations = self._extract_citations(parsed, retrieval)

        # Determine effective confidence (lowest of LLM + detector)
        llm_conf      = parsed.get("confidence", "medium")
        detector_conf = report.confidence_label
        effective_conf = self._lowest_confidence(llm_conf, detector_conf)

        return PlanIQResponse(
            query            = query,
            query_type       = query_type,
            answer_summary   = parsed.get("answer_summary", "No summary available."),
            full_answer      = parsed,
            citations        = citations,
            chunks_used      = parsed.get("chunks_used", []),
            jurisdiction     = jurisdiction,
            confidence       = effective_conf,
            confidence_score = report.confidence_score,
            disclaimer       = MANDATORY_DISCLAIMER,
            user_warning     = report.user_warning,
            requires_escalation = report.requires_escalation,
            latency_ms       = int((time.time() - start) * 1000),
            llm_provider     = self.provider_name,
            is_blocked       = report.is_blocked,
        )

    def _extract_citations(
        self, parsed: dict, retrieval: RetrievalResult
    ) -> list[dict]:
        """Extract citation metadata for UI rendering."""
        citations = []
        chunks_used = parsed.get("chunks_used", [])

        for chunk_num in chunks_used:
            idx = chunk_num - 1  # 1-indexed in prompt, 0-indexed in list
            if 0 <= idx < len(retrieval.chunks):
                chunk = retrieval.chunks[idx]
                citations.append({
                    "chunk_num":    chunk_num,
                    "source_title": chunk.source_title,
                    "section_ref":  chunk.section_ref,
                    "jurisdiction": chunk.jurisdiction,
                    "effective_date": chunk.effective_date,
                    "text_preview": chunk.text[:120] + "...",
                })
        return citations

    def _lowest_confidence(self, llm: str, detector: str) -> str:
        """Return the lower of two confidence labels."""
        order = {"high": 3, "medium": 2, "low": 1, "blocked": 0}
        llm_val      = order.get(llm, 2)
        detector_val = order.get(detector, 2)
        result_val   = min(llm_val, detector_val)
        return {v: k for k, v in order.items()}[result_val]

    # ── Private: special responses ────────────────

    def _blocked_response(
        self, query: str, report: HallucinationReport, start: float
    ) -> PlanIQResponse:
        block_flags = [f for f in report.flags if f.severity == "block"]
        reason = block_flags[0].message if block_flags else "Low confidence"
        return PlanIQResponse(
            query            = query,
            query_type       = "OTHER",
            answer_summary   = report.user_warning,
            full_answer      = {},
            confidence       = "blocked",
            confidence_score = report.confidence_score,
            disclaimer       = MANDATORY_DISCLAIMER,
            user_warning     = report.user_warning,
            requires_escalation = True,
            latency_ms       = int((time.time() - start) * 1000),
            llm_provider     = self.provider_name,
            is_blocked       = True,
            block_reason     = reason,
        )

    def _error_response(
        self, query: str, error: str, start: float
    ) -> PlanIQResponse:
        return PlanIQResponse(
            query            = query,
            query_type       = "OTHER",
            answer_summary   = "A technical error occurred. Please try again.",
            full_answer      = {"error": error},
            confidence       = "blocked",
            confidence_score = 0.0,
            disclaimer       = MANDATORY_DISCLAIMER,
            user_warning     = "Service temporarily unavailable.",
            latency_ms       = int((time.time() - start) * 1000),
            llm_provider     = self.provider_name,
            is_blocked       = True,
            block_reason     = f"LLM error: {error}",
        )

    def _log_response(self, response: PlanIQResponse) -> None:
        status = "[red]BLOCKED[/]" if response.is_blocked else "[green]OK[/]"
        console.log(
            f"[dim]Generated:[/] {status} | "
            f"type={response.query_type} | "
            f"confidence={response.confidence} | "
            f"latency={response.latency_ms}ms"
        )
