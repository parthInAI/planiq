"""
PlanIQ — Semantic Chunker
Splits Irish planning documents by logical rule/section boundaries.
NOT fixed-size token chunking — each chunk = one coherent planning rule.

Why this matters:
  Fixed chunking splits "Class 1 — Extension to the rear of a house not
  exceeding 40 sq.m" across two chunks. A split like that means the LLM
  sees the class name without the threshold, or the threshold without
  the class name. Both produce hallucinations.
"""

import re
import uuid
from datetime import date
from typing import Optional
from rich.console import Console
from schema import PlanningChunk, DocumentType, Jurisdiction, ConfidenceLevel

console = Console()


# ─────────────────────────────────────────────
#  Section boundary patterns for Irish planning docs
# ─────────────────────────────────────────────
SECTION_PATTERNS = [
    # Primary Act sections: "4.—(1)" or "Section 4." or "4. Development"
    re.compile(r'^(\d+[A-Z]?\.?—?\s*\(\d+\))', re.MULTILINE),
    # Schedule classes: "CLASS 1" or "Class 1 —"
    re.compile(r'^(CLASS\s+\d+[A-Z]?|Class\s+\d+[A-Z]?)', re.MULTILINE | re.IGNORECASE),
    # Numbered subsections in S.I.s: "Article 22" or "ARTICLE 22"
    re.compile(r'^(Article\s+\d+[A-Z]?)', re.MULTILINE | re.IGNORECASE),
    # Ministerial guideline chapters: "Chapter 3:" or "3. Design Standards"
    re.compile(r'^(\d+\.\s+[A-Z][a-zA-Z\s]+:?$)', re.MULTILINE),
    # Development plan chapters: "Chapter 7" or "7.1 Housing" or "Policy Objective H1"
    re.compile(r'^(?:Chapter\s+\d+|Policy\s+Objective\s+\w+|\d+\.\d+\s+[A-Z])', re.MULTILINE),
    # Single newline sections: "7.1.1 Standards" style
    re.compile(r'^(\d+\.\d+\.\d+\s+[A-Z])', re.MULTILINE),
    # Double newline (paragraph boundary — fallback for unstructured text)
    re.compile(r'\n\n+'),
]

# Chunk size controls
MIN_CHUNK_CHARS  = 80     # Lowered from 150 — dev plan paragraphs are often shorter
MAX_CHUNK_CHARS  = 2000   # Too long = retrieval precision loss
OVERLAP_CHARS    = 100    # Overlap between adjacent chunks to preserve context


class SemanticChunker:
    """
    Splits a planning document into PlanningChunk objects using
    section-aware boundary detection. Falls back to paragraph
    splitting when structural markers are absent.
    """

    def __init__(
        self,
        document_type:  DocumentType,
        jurisdiction:   Jurisdiction,
        source_title:   str,
        source_url:     str       = "",
        si_number:      str       = "",
        act_year:       Optional[int] = None,
        effective_date: Optional[date] = None,
        confidence:     ConfidenceLevel = ConfidenceLevel.HIGH,
        is_verbatim:    bool      = True,
    ):
        self.doc_id        = str(uuid.uuid4())
        self.document_type = document_type
        self.jurisdiction  = jurisdiction
        self.source_title  = source_title
        self.source_url    = source_url
        self.si_number     = si_number
        self.act_year      = act_year
        self.effective_date = effective_date
        self.confidence    = confidence
        self.is_verbatim   = is_verbatim

    def chunk(self, raw_text: str) -> list[PlanningChunk]:
        """
        Main entry point. Returns list of validated PlanningChunk objects.
        Each chunk has been through:
          1. Section boundary detection
          2. Size normalisation (merge short / split long)
          3. Section reference extraction
          4. Staleness metadata injection
          5. Pydantic validation
        """
        console.log(f"[bold cyan]Chunking:[/] {self.source_title[:60]}...")

        raw_segments = self._split_by_sections(raw_text)
        normalised   = self._normalise_sizes(raw_segments)
        chunks       = []

        for idx, segment in enumerate(normalised):
            segment = segment.strip()
            if len(segment) < MIN_CHUNK_CHARS:
                continue

            section_ref = self._extract_section_ref(segment)
            summary     = self._extract_summary(segment)

            chunk = PlanningChunk(
                chunk_id       = str(uuid.uuid4()),
                source_doc_id  = self.doc_id,
                chunk_index    = idx,
                text           = segment,
                summary        = summary,
                document_type  = self.document_type,
                jurisdiction   = self.jurisdiction,
                source_title   = self.source_title,
                source_url     = self.source_url,
                section_ref    = section_ref,
                si_number      = self.si_number,
                act_year       = self.act_year,
                effective_date = self.effective_date,
                confidence     = self.confidence,
                is_verbatim    = self.is_verbatim,
            )
            chunks.append(chunk)

        console.log(
            f"[green]✓[/] {len(chunks)} chunks from {len(raw_text):,} chars "
            f"| doc_id={self.doc_id[:8]}"
        )
        return chunks

    # ── private ──────────────────────────────

    def _split_by_sections(self, text: str) -> list[str]:
        """
        Try each section pattern in priority order.
        Use the first pattern that produces a reasonable number of splits.
        Fall back to paragraph splitting.
        """
        for pattern in SECTION_PATTERNS[:-1]:  # All but the paragraph fallback
            boundaries = [m.start() for m in pattern.finditer(text)]
            if len(boundaries) >= 3:
                return self._split_at_boundaries(text, boundaries)

        # Fallback: paragraph splitting
        paragraphs = SECTION_PATTERNS[-1].split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_at_boundaries(self, text: str, boundaries: list[int]) -> list[str]:
        """Split text at detected boundary positions with overlap."""
        segments = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            segment = text[start:end]
            # Add overlap from next segment to preserve cross-boundary context
            if i + 1 < len(boundaries) and OVERLAP_CHARS > 0:
                overlap_end = min(boundaries[i + 1] + OVERLAP_CHARS, len(text))
                segment = text[start:overlap_end]
            segments.append(segment)
        return segments

    def _normalise_sizes(self, segments: list[str]) -> list[str]:
        """
        Merge segments that are too short into their neighbour.
        Split segments that are too long at paragraph boundaries.
        """
        # Merge short segments into previous
        merged = []
        buffer = ""
        for seg in segments:
            if len(buffer) + len(seg) < MIN_CHUNK_CHARS:
                buffer += "\n\n" + seg
            else:
                if buffer:
                    merged.append(buffer)
                buffer = seg
        if buffer:
            merged.append(buffer)

        # Split long segments
        result = []
        for seg in merged:
            if len(seg) <= MAX_CHUNK_CHARS:
                result.append(seg)
            else:
                result.extend(self._split_long_segment(seg))
        return result

    def _split_long_segment(self, text: str) -> list[str]:
        """Split an oversized segment at paragraph boundaries."""
        parts   = re.split(r'\n\n+', text)
        chunks  = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= MAX_CHUNK_CHARS:
                current += ("\n\n" if current else "") + part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _extract_section_ref(self, text: str) -> str:
        """
        Pull the most specific section reference from the chunk text.
        Returned in structured format for the hallucination entity graph.
        """
        patterns = [
            # Schedule 2 Class references
            (r'(Class\s+\d+[A-Z]?\b)', "class"),
            # Section references with subsections
            (r'(section\s+\d+[A-Z]?\s*\(\d+\))', "section"),
            (r'(section\s+\d+[A-Z]?\b)', "section"),
            # Article references
            (r'(article\s+\d+[A-Z]?\b)', "article"),
            # S.I. references inline
            (r'(S\.I\.\s+No\.\s+\d+\s+of\s+\d{4})', "si"),
        ]
        refs = []
        for pattern, ref_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend(matches[:2])  # Max 2 per type to avoid noise

        # Prepend the SI number if present
        if self.si_number and self.si_number not in " ".join(refs):
            refs.insert(0, self.si_number)

        return " | ".join(refs[:4]) if refs else ""

    def _extract_summary(self, text: str) -> str:
        """
        Extract the first meaningful sentence as the chunk summary.
        Used by the cross-encoder reranker as context.
        """
        # Remove leading whitespace / newlines
        text = text.strip()
        # Take first non-empty line as summary, up to 120 chars
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 20:
                return line[:120] + ("..." if len(line) > 120 else "")
        return text[:120]
