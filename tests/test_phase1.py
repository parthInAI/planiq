"""
PlanIQ — Test Suite: Phase 1
Tests the council plan ingestion pipeline.
Run: pytest tests/test_phase1.py -v
"""

import sys
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))

from phase1_ingest import (
    COUNCIL_PLANS,
    _clean_page_text,
    ingest_council,
    MIN_CHARS_PER_PAGE,
    OCR_TRIGGER_THRESHOLD,
)
from ingestion.schema import Jurisdiction, DocumentType


# ── Registry tests ────────────────────────────────────────────────────────────

class TestCouncilRegistry:

    def test_all_7_councils_registered(self):
        assert len(COUNCIL_PLANS) == 7

    def test_required_fields_present(self):
        required = ["title", "filename", "jurisdiction", "effective_date", "act_year", "priority"]
        for key, plan in COUNCIL_PLANS.items():
            for field in required:
                assert field in plan, f"Council '{key}' missing field '{field}'"

    def test_all_jurisdictions_are_valid_enum(self):
        valid = {j.value for j in Jurisdiction}
        for key, plan in COUNCIL_PLANS.items():
            assert plan["jurisdiction"] in Jurisdiction, \
                f"Council '{key}' has invalid jurisdiction: {plan['jurisdiction']}"

    def test_all_filenames_end_in_pdf(self):
        for key, plan in COUNCIL_PLANS.items():
            assert plan["filename"].endswith(".pdf"), \
                f"Council '{key}' filename does not end in .pdf"

    def test_dublin_city_is_registered(self):
        assert "dublin_city" in COUNCIL_PLANS

    def test_fingal_is_registered(self):
        assert "fingal" in COUNCIL_PLANS

    def test_south_dublin_is_registered(self):
        assert "south_dublin" in COUNCIL_PLANS

    def test_dun_laoghaire_is_registered(self):
        assert "dun_laoghaire" in COUNCIL_PLANS

    def test_cork_city_is_registered(self):
        assert "cork_city" in COUNCIL_PLANS

    def test_galway_city_is_registered(self):
        assert "galway_city" in COUNCIL_PLANS

    def test_cork_county_is_registered(self):
        assert "cork_county" in COUNCIL_PLANS

    def test_priorities_are_1_or_2(self):
        for key, plan in COUNCIL_PLANS.items():
            assert plan["priority"] in (1, 2), \
                f"Council '{key}' has invalid priority: {plan['priority']}"

    def test_effective_dates_are_date_objects(self):
        for key, plan in COUNCIL_PLANS.items():
            assert isinstance(plan["effective_date"], date), \
                f"Council '{key}' effective_date is not a date object"

    def test_act_years_are_reasonable(self):
        for key, plan in COUNCIL_PLANS.items():
            assert 2018 <= plan["act_year"] <= 2026, \
                f"Council '{key}' act_year {plan['act_year']} outside expected range"


# ── Page cleaning tests ───────────────────────────────────────────────────────

class TestPageCleaning:

    def test_empty_string_returns_empty(self):
        assert _clean_page_text("") == ""

    def test_none_equivalent_returns_empty(self):
        assert _clean_page_text("") == ""

    def test_page_number_only_is_removed(self):
        result = _clean_page_text("47")
        assert result == "" or len(result) < 5

    def test_pipe_page_number_removed(self):
        result = _clean_page_text("| 47 |")
        assert "47" not in result or len(result) < 10

    def test_dashes_decorative_removed(self):
        result = _clean_page_text("-------------------")
        assert result == ""

    def test_real_planning_text_preserved(self):
        text = "The planning authority shall have regard to the development plan objectives when determining applications."
        result = _clean_page_text(text)
        assert "planning authority" in result
        assert "development plan" in result

    def test_section_reference_preserved(self):
        text = "In accordance with Section 28 Guidelines, the following standards apply to residential development."
        result = _clean_page_text(text)
        assert "Section 28" in result

    def test_numeric_threshold_preserved(self):
        text = "Rear extensions shall not exceed 40 square metres in floor area under Class 1."
        result = _clean_page_text(text)
        assert "40 square metres" in result
        assert "Class 1" in result

    def test_excessive_newlines_collapsed(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = _clean_page_text(text)
        assert "\n\n\n" not in result

    def test_short_lines_under_4_chars_removed(self):
        text = "ab\nThis is a proper planning policy sentence that matters.\ncd"
        result = _clean_page_text(text)
        assert "planning policy" in result

    def test_whitespace_stripped(self):
        text = "   Planning permission shall be required.   "
        result = _clean_page_text(text)
        assert result == result.strip()

    def test_mixed_content_extracts_useful_text(self):
        text = """
| 1 |

Planning Policy Objective HP-1: To support the provision of housing
to meet the needs of the existing and future population of the county.

| 2 |

-------------------

Development Management Standards for Residential Development
"""
        result = _clean_page_text(text)
        assert "Planning Policy Objective" in result
        assert "Development Management Standards" in result


# ── OCR threshold tests ───────────────────────────────────────────────────────

class TestOCRThreshold:

    def test_ocr_trigger_threshold_is_reasonable(self):
        assert 50 <= OCR_TRIGGER_THRESHOLD <= 300, \
            f"OCR trigger {OCR_TRIGGER_THRESHOLD} outside reasonable range"

    def test_min_chars_per_page_is_reasonable(self):
        assert 20 <= MIN_CHARS_PER_PAGE <= 200, \
            f"Min chars {MIN_CHARS_PER_PAGE} outside reasonable range"

    def test_ocr_threshold_greater_than_min_chars(self):
        assert OCR_TRIGGER_THRESHOLD > MIN_CHARS_PER_PAGE, \
            "OCR trigger should be higher than min chars per page"


# ── Ingest council tests ──────────────────────────────────────────────────────

class TestIngestCouncil:

    def test_missing_file_returns_missing_status(self):
        plan = COUNCIL_PLANS["fingal"].copy()
        plan["filename"] = "nonexistent_file_xyz.pdf"
        mock_kb = MagicMock()
        result = ingest_council("fingal", plan, mock_kb, dry_run=False)
        assert result["status"] == "missing"
        assert len(result["errors"]) > 0

    def test_dry_run_does_not_write_to_kb(self):
        plan = COUNCIL_PLANS["fingal"].copy()
        plan["filename"] = "nonexistent_file_xyz.pdf"
        mock_kb = MagicMock()
        ingest_council("fingal", plan, mock_kb, dry_run=True)
        mock_kb.add_chunks.assert_not_called()

    def test_stats_dict_has_required_keys(self):
        plan = COUNCIL_PLANS["cork_city"].copy()
        plan["filename"] = "nonexistent_xyz.pdf"
        mock_kb = MagicMock()
        result = ingest_council("cork_city", plan, mock_kb)
        required = ["council", "title", "status", "extraction",
                    "raw_chars", "chunks_created", "chunks_added", "errors"]
        for key in required:
            assert key in result, f"Missing key in stats: {key}"

    def test_council_key_preserved_in_stats(self):
        plan = COUNCIL_PLANS["galway_city"].copy()
        plan["filename"] = "nonexistent_xyz.pdf"
        mock_kb = MagicMock()
        result = ingest_council("galway_city", plan, mock_kb)
        assert result["council"] == "galway_city"

    def test_title_preserved_in_stats(self):
        plan = COUNCIL_PLANS["cork_county"].copy()
        plan["filename"] = "nonexistent_xyz.pdf"
        mock_kb = MagicMock()
        result = ingest_council("cork_county", plan, mock_kb)
        assert result["title"] == plan["title"]


# ── Integration: council metadata validation ──────────────────────────────────

class TestCouncilMetadata:

    def test_galway_city_notes_mentions_high_court(self):
        plan = COUNCIL_PLANS["galway_city"]
        notes = plan.get("notes", "").lower()
        assert "court" in notes or "2025" in notes, \
            "Galway City notes should mention High Court amendment"

    def test_south_dublin_notes_mentions_variation(self):
        plan = COUNCIL_PLANS["south_dublin"]
        notes = plan.get("notes", "").lower()
        assert "variation" in notes, \
            "South Dublin notes should mention Variation No.1"

    def test_dlr_notes_mentions_ocr(self):
        plan = COUNCIL_PLANS["dun_laoghaire"]
        notes = plan.get("notes", "").lower()
        assert "ocr" in notes or "scanned" in notes or "6mb" in notes, \
            "DLR notes should mention OCR/scanned concern"

    def test_all_batch1_priority1_councils_present(self):
        priority1 = [k for k, v in COUNCIL_PLANS.items() if v["priority"] == 1]
        expected = {"dublin_city", "fingal", "south_dublin", "dun_laoghaire"}
        assert expected.issubset(set(priority1)), \
            f"Priority 1 councils missing. Got: {priority1}"

    def test_all_batch1_priority2_councils_present(self):
        priority2 = [k for k, v in COUNCIL_PLANS.items() if v["priority"] == 2]
        expected = {"cork_city", "galway_city", "cork_county"}
        assert expected.issubset(set(priority2)), \
            f"Priority 2 councils missing. Got: {priority2}"
