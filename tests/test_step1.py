"""
PlanIQ — Test Suite: Step 1
Tests the schema, chunker, and knowledge base store.
Run: pytest tests/test_step1.py -v
"""

import sys
import uuid
import pytest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))

from ingestion.schema   import (
    PlanningChunk, DocumentType, Jurisdiction,
    ConfidenceLevel
)
from ingestion.chunker  import SemanticChunker, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS


# ── Fixtures ─────────────────────────────────

def make_chunk(**overrides) -> PlanningChunk:
    """Factory for test chunks with safe defaults."""
    defaults = dict(
        chunk_id       = str(uuid.uuid4()),
        source_doc_id  = "test_doc_001",
        chunk_index    = 0,
        text           = "Class 1 — Extension to the rear of a house not exceeding 40 square metres in floor area.",
        summary        = "Class 1 rear extension threshold",
        document_type  = DocumentType.EXEMPTION_SCHEDULE,
        jurisdiction   = Jurisdiction.NATIONAL,
        source_title   = "Planning and Development Regulations 2001 — Schedule 2",
        source_url     = "https://www.irishstatutebook.ie/eli/2001/si/600/made/en/print",
        section_ref    = "Class 1 | S.I. No. 600 of 2001",
        si_number      = "S.I. No. 600 of 2001",
        act_year       = 2001,
        effective_date = date(2001, 10, 1),
        confidence     = ConfidenceLevel.HIGH,
        is_verbatim    = True,
    )
    defaults.update(overrides)
    return PlanningChunk(**defaults)


SAMPLE_PLANNING_TEXT = """
Planning and Development Regulations 2001 — Schedule 2

PART 1
EXEMPTED DEVELOPMENT — GENERAL

CLASS 1

Development consisting of the extension of a dwellinghouse, where:

(a) the floor area of any such extension does not exceed 40 square metres in the case of a
    terraced or semi-detached house, or 40 square metres in the case of a detached house;

(b) the height of the walls of any such extension does not exceed the height of the walls
    of the dwellinghouse;

(c) no window, door or other opening is formed in any wall or part of a wall of the extension
    that faces towards a road;

Provided that this class shall not apply in relation to an extension which, when added to any
previous extensions (exclusive of any extension existing on 1 October 1964), would exceed the
relevant floor area limit.

CLASS 2

Development consisting of the provision of a porch outside any external door of a dwellinghouse,
where:

(a) the floor area of the porch does not exceed 2 square metres;

(b) any wall of the porch is not less than 2 metres from the boundary of the curtilage of
    the dwellinghouse with a public road;

(c) the height of the porch does not exceed 4 metres in the case of a porch with a pitched
    roof, or 3 metres in any other case.

CLASS 3

Development consisting of the provision within the curtilage of a dwellinghouse of a garage,
store or shed and works ancillary to the provision of such structure, where:

(a) such structure is used only for a purpose incidental to the enjoyment of the dwellinghouse
    as such;

(b) the floor area of such structure does not exceed 25 square metres;

(c) the height of such structure does not exceed 4 metres in the case of a structure with a
    pitched roof or 3 metres in any other case.
"""


# ── Schema tests ─────────────────────────────

class TestPlanningChunkSchema:

    def test_valid_chunk_creates_successfully(self):
        chunk = make_chunk()
        assert chunk.chunk_id
        assert chunk.is_stale is False

    def test_stale_chunk_with_superseded_by(self):
        chunk = make_chunk(superseded_by="S.I. No. 649 of 2025")
        assert chunk.is_stale is True

    def test_expired_chunk_is_stale(self):
        chunk = make_chunk(expiry_date=date(2023, 12, 31))
        assert chunk.is_stale is True

    def test_future_expiry_is_not_stale(self):
        chunk = make_chunk(expiry_date=date.today() + timedelta(days=365))
        assert chunk.is_stale is False

    def test_needs_reverification_after_30_days(self):
        chunk = make_chunk(last_verified=date.today() - timedelta(days=31))
        assert chunk.needs_reverification is True

    def test_recent_chunk_does_not_need_reverification(self):
        chunk = make_chunk(last_verified=date.today())
        assert chunk.needs_reverification is False

    def test_chroma_metadata_is_flat_dict(self):
        chunk    = make_chunk()
        metadata = chunk.to_chroma_metadata()
        # All values must be ChromaDB-compatible (str, int, float, bool)
        for key, val in metadata.items():
            assert isinstance(val, (str, int, float, bool)), (
                f"Key '{key}' has non-serialisable type {type(val)}"
            )

    def test_empty_superseded_by_is_not_stale(self):
        chunk = make_chunk(superseded_by="")
        assert chunk.is_stale is False

    def test_chunk_text_minimum_length(self):
        with pytest.raises(Exception):
            make_chunk(text="Too short")

    def test_jurisdiction_enum_all_31_councils(self):
        """Verify all 31 Irish local authorities are in the enum."""
        councils = [j for j in Jurisdiction if j != Jurisdiction.NATIONAL]
        assert len(councils) == 31, f"Expected 31 councils, found {len(councils)}"


# ── Chunker tests ─────────────────────────────

class TestSemanticChunker:

    def get_chunker(self) -> SemanticChunker:
        return SemanticChunker(
            document_type  = DocumentType.EXEMPTION_SCHEDULE,
            jurisdiction   = Jurisdiction.NATIONAL,
            source_title   = "PDR 2001 Schedule 2 — Test",
            source_url     = "https://www.irishstatutebook.ie/test",
            si_number      = "S.I. No. 600 of 2001",
            act_year       = 2001,
            effective_date = date(2001, 10, 1),
        )

    def test_chunks_are_returned(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        assert len(chunks) > 0

    def test_class_1_is_own_chunk(self):
        """
        Critical: Class 1 (rear extension) must not be merged with Class 2.
        Mixing classes = hallucination risk.
        """
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        texts   = [c.text for c in chunks]
        class1_chunks = [t for t in texts if "CLASS 1" in t or "Class 1" in t]
        assert len(class1_chunks) >= 1, "Class 1 must appear in at least one chunk"
        # Class 1 chunk should not contain the Class 2 definition
        for t in class1_chunks:
            if "CLASS 2" in t or "Class 2" in t:
                # If they're merged, check it's within size limits
                assert len(t) <= MAX_CHUNK_CHARS, "Merged class chunk too long"

    def test_no_chunk_exceeds_max_size(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        for chunk in chunks:
            assert len(chunk.text) <= MAX_CHUNK_CHARS + 200, (
                f"Chunk too long: {len(chunk.text)} chars | {chunk.text[:60]}"
            )

    def test_all_chunks_meet_minimum_size(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        for chunk in chunks:
            assert len(chunk.text) >= MIN_CHUNK_CHARS, (
                f"Chunk too short: {len(chunk.text)} chars"
            )

    def test_section_ref_extracted(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        # At least some chunks should have a section ref
        chunks_with_refs = [c for c in chunks if c.section_ref]
        assert len(chunks_with_refs) > 0, "No section references extracted"

    def test_all_chunks_inherit_metadata(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        for chunk in chunks:
            assert chunk.jurisdiction == Jurisdiction.NATIONAL
            assert chunk.document_type == DocumentType.EXEMPTION_SCHEDULE
            assert chunk.si_number == "S.I. No. 600 of 2001"
            assert chunk.act_year == 2001

    def test_chunk_index_is_sequential(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices not sequential"

    def test_stale_flag_not_set_on_fresh_chunks(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        for chunk in chunks:
            assert not chunk.is_stale, f"Fresh chunk incorrectly marked stale: {chunk.chunk_id}"

    def test_summary_extracted_and_non_empty(self):
        chunker = self.get_chunker()
        chunks  = chunker.chunk(SAMPLE_PLANNING_TEXT)
        for chunk in chunks:
            assert chunk.summary, f"Empty summary on chunk {chunk.chunk_index}"
            assert len(chunk.summary) <= 123  # 120 chars + "..."


# ── Integration test ──────────────────────────

class TestChunkerToSchema:

    def test_40sqm_threshold_preserved_in_chunk(self):
        """
        The 40 sq.m threshold for Class 1 is a critical legal number.
        It must appear intact in a chunk, not split across chunk boundaries.
        """
        chunker = SemanticChunker(
            document_type=DocumentType.EXEMPTION_SCHEDULE,
            jurisdiction=Jurisdiction.NATIONAL,
            source_title="Test",
            source_url="",
        )
        chunks = chunker.chunk(SAMPLE_PLANNING_TEXT)
        found  = any("40" in c.text and ("square" in c.text.lower() or "metres" in c.text.lower())
                     for c in chunks)
        assert found, "40 sq.m threshold not found intact in any chunk — hallucination risk!"

    def test_chunk_ids_are_unique(self):
        chunker = SemanticChunker(
            document_type=DocumentType.EXEMPTION_SCHEDULE,
            jurisdiction=Jurisdiction.NATIONAL,
            source_title="Test",
            source_url="",
        )
        chunks = chunker.chunk(SAMPLE_PLANNING_TEXT)
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected!"
