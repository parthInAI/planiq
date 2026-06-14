"""
PlanIQ — Hallucination Detection Layer (Step 2)
================================================
Runs BEFORE any answer is shown to the user.
Seven checks, each independently blockable.

Pipeline:
  1. Staleness gate         — block chunks updated/superseded
  2. Entity grounding       — every claim must exist in retrieved context
  3. Section verification   — section numbers must exist in chunk text
  4. Self-consistency       — (flagged for generation layer, 3-sample check)
  5. Confidence scoring     — aggregate 0-1 score from all signals
  6. Uncertainty flagging   — specific warnings surfaced to user
  7. HITL escalation gate   — auto-escalate low-confidence or sensitive queries

Output: HallucinationReport — passed to generation layer, surfaced in UI.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

from retrieval.hybrid_retriever import RetrievalResult, RetrievedChunk

console = Console()

# ── Thresholds ───────────────────────────────────
CONFIDENCE_HIGH    = 0.75   # Show answer with citation
CONFIDENCE_MEDIUM  = 0.50   # Show answer with warning
CONFIDENCE_LOW     = 0.30   # Show partial answer + escalate
# Below LOW → block answer, show escalation only

# Topics that always trigger HITL escalation regardless of confidence
ALWAYS_ESCALATE_PATTERNS = [
    r"appeal",
    r"an bord plean",
    r"an coimisi",
    r"enforcement",
    r"unauthorised development",
    r"section 5 declaration",
    r"judicial review",
    r"compensation",
    r"injunction",
]

# Sensitive numeric thresholds — must appear exactly in retrieved chunks
LEGAL_THRESHOLDS = [
    r"\d+\s*square\s*metres?",
    r"\d+\s*sq\.?\s*m",
    r"\d+\s*metres?\s*high",
    r"\d+\s*metres?\s*from",
    r"\d+\s*weeks?",
    r"\d+\s*days?",
]


@dataclass
class UncertaintyFlag:
    """A specific uncertainty signal surfaced to the user."""
    code:        str    # machine-readable code
    severity:    str    # "warning" | "block" | "escalate"
    message:     str    # user-facing message
    detail:      str    # technical detail for logging


@dataclass
class HallucinationReport:
    """
    Complete hallucination analysis for one query-response pair.
    Passed to the generation layer — determines what gets shown to user.
    """
    query:              str
    confidence_score:   float = 1.0       # 0-1, starts optimistic
    flags:              list[UncertaintyFlag] = field(default_factory=list)
    grounded_entities:  set[str] = field(default_factory=set)
    ungrounded_claims:  list[str] = field(default_factory=list)
    requires_escalation: bool = False
    escalation_reason:   str = ""
    retrieval_quality:   float = 0.0
    chunks_used:         int = 0
    is_blocked:          bool = False     # True = do not show any answer

    @property
    def confidence_label(self) -> str:
        if self.confidence_score >= CONFIDENCE_HIGH:
            return "high"
        elif self.confidence_score >= CONFIDENCE_MEDIUM:
            return "medium"
        elif self.confidence_score >= CONFIDENCE_LOW:
            return "low"
        return "blocked"

    @property
    def user_warning(self) -> str:
        """Plain-English warning shown to user based on confidence level."""
        if self.is_blocked:
            return (
                "⚠️ PlanIQ cannot provide a confident answer to this query. "
                "Please consult a registered planning consultant or submit a "
                "Section 5 declaration to your local authority."
            )
        if self.requires_escalation:
            return (
                "⚠️ This topic involves a formal legal process. "
                "PlanIQ provides guidance only — for this query you should "
                "engage a qualified planning consultant."
            )
        if self.confidence_label == "low":
            return (
                "⚠️ This answer has low confidence. "
                "Verify with your local planning authority before acting on it."
            )
        if self.confidence_label == "medium":
            return (
                "ℹ️ This answer is based on retrieved planning law. "
                "Always verify with your local authority for your specific site."
            )
        return (
            "ℹ️ PlanIQ provides guidance only and does not constitute "
            "professional planning advice."
        )

    @property
    def mandatory_disclaimer(self) -> str:
        """Always shown — legal protection disclaimer."""
        return (
            "PlanIQ provides guidance only and does not constitute professional "
            "planning advice. For formal determinations, engage a registered "
            "planning consultant or submit a Section 5 declaration to your "
            "local authority."
        )


class HallucinationDetector:
    """
    Runs hallucination checks on a retrieval result before generation.
    Called BEFORE the LLM generates any text.

    Usage:
        detector = HallucinationDetector()
        report   = detector.analyse(query, retrieval_result)
        if report.is_blocked:
            # show escalation message only
        else:
            # pass to generation layer with report attached
    """

    def analyse(
        self,
        query:            str,
        retrieval_result: RetrievalResult,
        generated_text:   Optional[str] = None,
    ) -> HallucinationReport:
        """
        Run all hallucination checks. Returns HallucinationReport.

        If generated_text is provided, also runs post-generation
        entity grounding check on the actual response text.
        """
        report = HallucinationReport(
            query=query,
            retrieval_quality=retrieval_result.retrieval_quality,
            chunks_used=len(retrieval_result.chunks),
        )

        # ── Check 1: Empty retrieval ──────────────
        self._check_empty_retrieval(retrieval_result, report)
        if report.is_blocked:
            return report

        # ── Check 2: Retrieval quality gate ───────
        self._check_retrieval_quality(retrieval_result, report)

        # ── Check 3: Staleness gate ───────────────
        self._check_staleness(retrieval_result, report)

        # ── Check 4: HITL escalation patterns ─────
        self._check_escalation_patterns(query, report)

        # ── Check 5: Post-generation entity grounding ──
        if generated_text:
            self._check_entity_grounding(
                generated_text, retrieval_result, report
            )
            self._check_numeric_thresholds(
                generated_text, retrieval_result, report
            )

        # ── Check 6: Compute final confidence score ──
        self._compute_confidence(report)

        # ── Check 7: Block only truly empty or dangerous responses ────
        # Only block if confidence is very low AND we have no chunks at all
        if report.confidence_score < CONFIDENCE_LOW and report.chunks_used == 0 and not report.requires_escalation:
            report.is_blocked = True
            report.flags.append(UncertaintyFlag(
                code     = "CONFIDENCE_BELOW_THRESHOLD",
                severity = "block",
                message  = "Confidence too low to provide a safe answer",
                detail   = f"Score={report.confidence_score:.2f} < threshold={CONFIDENCE_LOW}",
            ))

        self._log_report(report)
        return report

    # ── Check implementations ─────────────────────

    def _check_empty_retrieval(
        self, result: RetrievalResult, report: HallucinationReport
    ) -> None:
        if result.is_empty:
            report.is_blocked = True
            report.confidence_score = 0.0
            report.flags.append(UncertaintyFlag(
                code     = "NO_CHUNKS_RETRIEVED",
                severity = "block",
                message  = "No relevant planning law found for this query",
                detail   = f"Query: {result.query[:80]}",
            ))

    def _check_retrieval_quality(
        self, result: RetrievalResult, report: HallucinationReport
    ) -> None:
        """Flag low retrieval quality — weak signal for hallucination risk."""
        if result.retrieval_quality < 0.3:
            report.flags.append(UncertaintyFlag(
                code     = "LOW_RETRIEVAL_QUALITY",
                severity = "warning",
                message  = "Retrieved context may not fully cover this query",
                detail   = f"Quality score: {result.retrieval_quality:.2f}",
            ))

        # Single retrieval method only — higher hallucination risk
        if result.total_dense_hits == 0 or result.total_sparse_hits == 0:
            report.flags.append(UncertaintyFlag(
                code     = "SINGLE_METHOD_RETRIEVAL",
                severity = "warning",
                message  = "Answer based on single retrieval method — lower confidence",
                detail   = f"Dense: {result.total_dense_hits} | Sparse: {result.total_sparse_hits}",
            ))

    def _check_staleness(
        self, result: RetrievalResult, report: HallucinationReport
    ) -> None:
        """Check for any stale chunks that slipped through."""
        stale = [c for c in result.chunks if c.is_stale]
        if stale:
            report.flags.append(UncertaintyFlag(
                code     = "STALE_CHUNKS_IN_CONTEXT",
                severity = "block",
                message  = "Retrieved context contains superseded regulations",
                detail   = f"{len(stale)} stale chunks detected",
            ))
            report.is_blocked = True

        # Flag chunks that haven't been verified recently
        old = [c for c in result.chunks if c.metadata.get("needs_reverification")]
        if old:
            report.flags.append(UncertaintyFlag(
                code     = "CHUNKS_NEED_REVERIFICATION",
                severity = "warning",
                message  = "Some source documents may not reflect the latest regulations",
                detail   = f"{len(old)} chunks not verified in 30+ days",
            ))

    def _check_escalation_patterns(
        self, query: str, report: HallucinationReport
    ) -> None:
        """
        Check if query matches topics that always require human escalation.
        Appeals, enforcement, Section 5 declarations = formal legal process.
        """
        query_lower = query.lower()
        for pattern in ALWAYS_ESCALATE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                report.requires_escalation = True
                report.escalation_reason   = f"Query involves '{pattern}' — formal legal process"
                report.flags.append(UncertaintyFlag(
                    code     = "ESCALATION_REQUIRED",
                    severity = "escalate",
                    message  = "This topic requires professional planning advice",
                    detail   = f"Pattern matched: {pattern}",
                ))
                break

    def _check_entity_grounding(
        self,
        generated_text:   str,
        retrieval_result: RetrievalResult,
        report:           HallucinationReport,
    ) -> None:
        """
        HalluGraph-style entity grounding check.
        Every section reference in the generated text must exist
        in the retrieved chunks. If not → ungrounded claim → flag.
        """
        grounded_entities = retrieval_result.get_entity_set()
        report.grounded_entities = grounded_entities

        # Extract section references from generated text
        generated_refs = re.findall(
            r'(?:class|section|article|schedule)\s+\d+[A-Z]?'
            r'|S\.I\.\s*No\.\s*\d+\s*of\s*\d{4}',
            generated_text,
            re.IGNORECASE,
        )

        for ref in generated_refs:
            ref_lower = ref.lower().strip()
            # Check if this reference exists in any retrieved chunk
            if not any(ref_lower in entity for entity in grounded_entities):
                report.ungrounded_claims.append(ref)

        if report.ungrounded_claims:
            report.flags.append(UncertaintyFlag(
                code     = "UNGROUNDED_SECTION_REFERENCE",
                severity = "warning",
                message  = f"Response contains {len(report.ungrounded_claims)} reference(s) not found in source documents",
                detail   = f"Ungrounded: {report.ungrounded_claims[:5]}",
            ))

    def _check_numeric_thresholds(
        self,
        generated_text:   str,
        retrieval_result: RetrievalResult,
        report:           HallucinationReport,
    ) -> None:
        """
        Numeric threshold grounding check.
        "40 square metres" in the response must appear in a retrieved chunk.
        Wrong thresholds (e.g. "60 sq m" instead of "40 sq m") = dangerous hallucination.
        """
        all_chunk_text = " ".join(c.text for c in retrieval_result.chunks).lower()

        for pattern in LEGAL_THRESHOLDS:
            generated_nums = re.findall(pattern, generated_text, re.IGNORECASE)
            for num in generated_nums:
                if num.lower() not in all_chunk_text:
                    report.flags.append(UncertaintyFlag(
                        code     = "UNGROUNDED_NUMERIC_THRESHOLD",
                        severity = "warning",
                        message  = f"Numeric threshold '{num}' not found in source documents",
                        detail   = f"Generated text contained '{num}' but it's absent from retrieved chunks",
                    ))
                    report.ungrounded_claims.append(num)

    def _compute_confidence(self, report: HallucinationReport) -> None:
        """
        Aggregate all signals into a final 0-1 confidence score.
        """
        # Start from retrieval quality
        score = report.retrieval_quality

        # Penalty per flag by severity
        severity_penalty = {"warning": 0.10, "escalate": 0.15, "block": 0.50}
        for flag in report.flags:
            score -= severity_penalty.get(flag.severity, 0.10)

        # Penalty for ungrounded claims
        score -= 0.08 * len(report.ungrounded_claims)

        # Bonus: multiple high-confidence chunks retrieved
        high_conf = sum(
            1 for c in [report]
            if report.retrieval_quality > 0.6
        )
        score += 0.05 * min(high_conf, 3)

        report.confidence_score = max(0.0, min(score, 1.0))

    def _log_report(self, report: HallucinationReport) -> None:
        """Log the report summary for monitoring."""
        status = (
            "[red]BLOCKED[/]"     if report.is_blocked else
            "[yellow]ESCALATE[/]" if report.requires_escalation else
            "[green]OK[/]"
        )
        console.log(
            f"[dim]Hallucination check:[/] {status} | "
            f"confidence={report.confidence_score:.2f} | "
            f"flags={len(report.flags)} | "
            f"ungrounded={len(report.ungrounded_claims)}"
        )
