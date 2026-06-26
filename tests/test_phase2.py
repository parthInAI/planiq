"""
PlanIQ — Test Suite: Phase 2 — Document Review Tool
Tests the PDF field extractor, Article 22 checker, and /upload endpoint.
Run: pytest tests/test_phase2.py -v
"""

import sys
import pytest
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "document_review"))

from document_review.article22_checker import (
    Article22Checker, CheckStatus, GapReport, CheckResult,
    ARTICLE_22_REQUIREMENTS,
)
from document_review.pdf_extractor import PDFFieldExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

def complete_fields(**overrides) -> dict:
    """A fully compliant planning application field set."""
    fields = {
        "has_application_form":       True,
        "applicant_name":             "John Murphy",
        "planning_authority":         "Dublin City Council",
        "development_description":    "Construction of a single storey rear extension to existing dwelling.",
        "permission_type":            "permission",
        "form_signed":                True,
        "has_os_map":                 True,
        "site_outlined_red":          True,
        "os_map_scale":               "1:1000",
        "has_site_layout_plan":       True,
        "site_layout_scale":          "1:500",
        "has_floor_plans":            True,
        "floor_plan_scale":           "1:200",
        "has_elevations":             True,
        "has_sections":               True,
        "drawings_coloured":          True,
        "scales_consistent":          True,
        "is_extension_or_alteration": True,
        "has_newspaper_notice":       True,
        "newspaper_name":             "Irish Times",
        "newspaper_publication_date": (date.today() - timedelta(days=7)).isoformat(),
        "application_lodgement_date": date.today().isoformat(),
        "has_site_notice":            True,
        "applicant_is_owner":         True,
        "has_landowner_consent":      False,
        "has_fee_included":           True,
        "fee_amount":                 125.0,
        "gross_floor_area_sqm":       30.0,
        "is_protected_structure":     False,
        "is_aca":                     False,
        "has_heritage_photographs":   False,
        "has_heritage_particulars":   False,
        "raw_text_length":            5000,
        "extraction_confidence":      "high",
    }
    fields.update(overrides)
    return fields


# ── Article 22 Registry Tests ─────────────────────────────────────────────────

class TestArticle22Registry:

    def test_all_required_keys_present(self):
        required_keys = [
            "22_2_a_form", "22_2_a_signature", "22_2_b_os_map",
            "22_2_c_site_layout", "22_2_d_floor_plans", "22_2_d_elevations",
            "22_2_d_scale_consistency", "22_2_e_written_statement",
            "22_2_f_newspaper_notice", "22_newspaper_timing",
            "19_site_notice", "22_2_g_legal_interest", "22_2_h_fee",
            "22a_protected_structure",
        ]
        for key in required_keys:
            assert key in ARTICLE_22_REQUIREMENTS, f"Missing requirement: {key}"

    def test_all_requirements_have_article_reference(self):
        for key, req in ARTICLE_22_REQUIREMENTS.items():
            assert "article" in req, f"{key} missing article reference"
            assert req["article"].startswith("Article"), f"{key} article should start with 'Article'"

    def test_all_requirements_have_severity(self):
        for key, req in ARTICLE_22_REQUIREMENTS.items():
            assert req["severity"] in ("critical", "major", "minor"), \
                f"{key} has invalid severity: {req['severity']}"

    def test_critical_items_include_form_and_maps(self):
        critical = [k for k, v in ARTICLE_22_REQUIREMENTS.items()
                    if v["severity"] == "critical"]
        assert "22_2_a_form" in critical
        assert "22_2_b_os_map" in critical
        assert "22_2_c_site_layout" in critical
        assert "22_newspaper_timing" in critical


# ── Article 22 Checker Tests ──────────────────────────────────────────────────

class TestArticle22Checker:

    def setup_method(self):
        self.checker = Article22Checker()

    def test_complete_application_passes(self):
        report = self.checker.check(complete_fields())
        assert report.failed == 0
        assert report.overall_status in ("valid", "review_required")

    def test_missing_form_fails(self):
        fields = complete_fields(has_application_form=False, applicant_name="")
        report = self.checker.check(fields)
        form_check = next(c for c in report.checks if "Form No. 1" in c.item)
        assert form_check.status in (CheckStatus.MISSING, CheckStatus.WARNING)

    def test_unsigned_form_fails(self):
        fields = complete_fields(form_signed=False)
        report = self.checker.check(fields)
        sig_check = next(c for c in report.checks if "signature" in c.item.lower())
        assert sig_check.status == CheckStatus.FAIL

    def test_missing_os_map_fails(self):
        fields = complete_fields(has_os_map=False)
        report = self.checker.check(fields)
        os_check = next(c for c in report.checks if "Ordnance Survey" in c.item)
        assert os_check.status == CheckStatus.MISSING

    def test_wrong_scale_site_layout_fails(self):
        fields = complete_fields(site_layout_scale="1:1000")
        report = self.checker.check(fields)
        layout_check = next(c for c in report.checks if "1:500" in c.item)
        assert layout_check.status == CheckStatus.FAIL

    def test_correct_scale_site_layout_passes(self):
        fields = complete_fields(site_layout_scale="1:500")
        report = self.checker.check(fields)
        layout_check = next(c for c in report.checks if "1:500" in c.item)
        assert layout_check.status == CheckStatus.PASS

    def test_newspaper_within_14_days_passes(self):
        fields = complete_fields(
            newspaper_publication_date=(date.today() - timedelta(days=10)).isoformat(),
            application_lodgement_date=date.today().isoformat(),
        )
        report = self.checker.check(fields)
        timing = next(c for c in report.checks if "2 weeks" in c.item)
        assert timing.status == CheckStatus.PASS

    def test_newspaper_beyond_14_days_fails(self):
        fields = complete_fields(
            newspaper_publication_date=(date.today() - timedelta(days=20)).isoformat(),
            application_lodgement_date=date.today().isoformat(),
        )
        report = self.checker.check(fields)
        timing = next(c for c in report.checks if "2 weeks" in c.item)
        assert timing.status == CheckStatus.FAIL

    def test_newspaper_before_application_fails(self):
        fields = complete_fields(
            newspaper_publication_date=date.today().isoformat(),
            application_lodgement_date=(date.today() - timedelta(days=3)).isoformat(),
        )
        report = self.checker.check(fields)
        timing = next(c for c in report.checks if "2 weeks" in c.item)
        assert timing.status == CheckStatus.FAIL

    def test_missing_site_notice_flagged(self):
        fields = complete_fields(has_site_notice=False)
        report = self.checker.check(fields)
        notice = next(c for c in report.checks if "Site notice" in c.item)
        assert notice.status == CheckStatus.MISSING

    def test_non_owner_without_consent_fails(self):
        fields = complete_fields(applicant_is_owner=False, has_landowner_consent=False)
        report = self.checker.check(fields)
        legal = next(c for c in report.checks if "legal interest" in c.item.lower())
        assert legal.status == CheckStatus.FAIL

    def test_non_owner_with_consent_passes(self):
        fields = complete_fields(applicant_is_owner=False, has_landowner_consent=True)
        report = self.checker.check(fields)
        legal = next(c for c in report.checks if "legal interest" in c.item.lower())
        assert legal.status == CheckStatus.PASS

    def test_missing_newspaper_notice_flagged(self):
        fields = complete_fields(has_newspaper_notice=False)
        report = self.checker.check(fields)
        notice = next(c for c in report.checks if "Newspaper notice" in c.item)
        assert notice.status == CheckStatus.MISSING

    def test_protected_structure_triggers_extra_check(self):
        fields = complete_fields(
            is_protected_structure=True,
            has_heritage_photographs=False,
            has_heritage_particulars=False,
        )
        report = self.checker.check(fields)
        heritage = next((c for c in report.checks if "protected structure" in c.item.lower()), None)
        assert heritage is not None
        assert heritage.status == CheckStatus.FAIL

    def test_protected_structure_with_photos_passes(self):
        fields = complete_fields(
            is_protected_structure=True,
            has_heritage_photographs=True,
            has_heritage_particulars=True,
        )
        report = self.checker.check(fields)
        heritage = next(c for c in report.checks if "protected structure" in c.item.lower())
        assert heritage.status == CheckStatus.PASS

    def test_non_protected_structure_no_heritage_check(self):
        fields = complete_fields(is_protected_structure=False, is_aca=False)
        report = self.checker.check(fields)
        heritage = [c for c in report.checks if "protected structure" in c.item.lower()]
        assert len(heritage) == 0

    def test_overall_status_likely_invalid_when_fails(self):
        fields = complete_fields(form_signed=False, has_os_map=False, has_site_notice=False)
        report = self.checker.check(fields)
        assert report.overall_status == "likely_invalid"

    def test_overall_status_valid_when_all_pass(self):
        report = self.checker.check(complete_fields())
        assert report.overall_status in ("valid", "review_required")

    def test_gap_report_counts_correct(self):
        fields = complete_fields(form_signed=False, has_os_map=False)
        report = self.checker.check(fields)
        total = report.passed + report.failed + report.warnings + report.missing
        assert total == report.total_checks

    def test_gap_report_has_disclaimer(self):
        report = self.checker.check(complete_fields())
        assert len(report.disclaimer) > 50
        assert "professional" in report.disclaimer.lower()

    def test_check_results_have_all_fields(self):
        report = self.checker.check(complete_fields())
        for check in report.checks:
            assert check.article
            assert check.item
            assert check.status in CheckStatus
            assert check.finding
            assert check.requirement
            assert check.action
            assert check.severity in ("critical", "major", "minor")

    def test_scale_validator_accepts_correct_scales(self):
        checker = Article22Checker()
        assert checker._scale_ok("1:500", min_urban=500) is True
        assert checker._scale_ok("1:200", min_urban=500) is True
        assert checker._scale_ok("1:100", min_urban=500) is True

    def test_scale_validator_rejects_too_small(self):
        checker = Article22Checker()
        assert checker._scale_ok("1:1000", min_urban=500) is False
        assert checker._scale_ok("1:2500", min_urban=500) is False

    def test_scale_validator_handles_unparseable(self):
        checker = Article22Checker()
        assert checker._scale_ok("not a scale", min_urban=500) is True


# ── PDF Extractor Tests ───────────────────────────────────────────────────────

class TestPDFFieldExtractor:

    def setup_method(self):
        self.extractor = PDFFieldExtractor()

    def test_empty_bytes_returns_fields_dict(self):
        fields = self.extractor.extract(b"")
        assert isinstance(fields, dict)
        assert "raw_text_length" in fields

    def test_non_pdf_bytes_returns_fields_dict(self):
        fields = self.extractor.extract(b"not a real pdf")
        assert isinstance(fields, dict)

    def test_scale_pattern_extracts_1_200(self):
        text = "Floor plans drawn to a scale of 1:200 as required."
        scale = self.extractor._extract_floor_plan_scale(text)
        assert "200" in scale

    def test_scale_pattern_extracts_1_500(self):
        text = "Site layout plan at 1:500 scale showing site boundary."
        scale = self.extractor._extract_site_layout_scale(text)
        assert "500" in scale

    def test_detects_application_form(self):
        text = "planning application form applicant planning authority nature of application development"
        result = self.extractor._detect_form(text)
        assert result is True

    def test_detects_floor_plans(self):
        text = "ground floor plan first floor plan drawn at 1:100 scale"
        assert self.extractor._detect_floor_plans(text.lower()) is True

    def test_detects_elevations(self):
        text = "north elevation south elevation east elevation west elevation"
        assert self.extractor._detect_elevations(text.lower()) is True

    def test_detects_os_map(self):
        text = "ordnance survey location map at 1:1000 scale site outlined in red"
        assert self.extractor._detect_os_map(text.lower()) is True

    def test_detects_newspaper_notice(self):
        text = "newspaper notice published in the irish times on 15 june 2026"
        assert self.extractor._detect_newspaper_notice(text.lower()) is True

    def test_detects_site_notice(self):
        text = "site notice erected at the entrance to the site"
        assert self.extractor._detect_site_notice(text.lower()) is True

    def test_detects_protected_structure(self):
        text = "the proposed development affects a protected structure rps reference"
        assert self.extractor._detect_protected_structure(text.lower()) is True

    def test_detects_aca(self):
        text = "site is located within an architectural conservation area aca"
        assert self.extractor._detect_aca(text.lower()) is True

    def test_detects_extension(self):
        text = "proposed rear extension to existing dwelling"
        assert self.extractor._detect_extension(text.lower()) is True

    def test_extracts_planning_authority_dublin_city(self):
        text = "To: Dublin City Council Planning Department"
        result = self.extractor._extract_planning_authority(text)
        assert "Dublin City" in result

    def test_extracts_planning_authority_cork(self):
        text = "Application to Cork City Council for planning permission"
        result = self.extractor._extract_planning_authority(text)
        assert "Cork" in result

    def test_confidence_low_for_short_text(self):
        text = "very short text"
        score = self.extractor._score_confidence(text)
        assert score == "low"

    def test_confidence_high_for_planning_document(self):
        text = ("planning permission application development council "
                "applicant scale elevation site permission application "
                "council permission planning development ") * 10
        score = self.extractor._score_confidence(text)
        assert score in ("medium", "high")

    def test_normalise_date_dd_mm_yyyy(self):
        result = self.extractor._normalise_date("15/06/2026")
        assert result == "2026-06-15"

    def test_normalise_date_invalid_returns_none(self):
        result = self.extractor._normalise_date("not a date")
        assert result is None

    def test_all_expected_keys_in_output(self):
        fields = self.extractor.extract(b"")
        expected_keys = [
            "has_application_form", "applicant_name", "planning_authority",
            "development_description", "form_signed", "has_os_map",
            "has_site_layout_plan", "has_floor_plans", "has_elevations",
            "has_newspaper_notice", "has_site_notice", "applicant_is_owner",
            "has_fee_included", "is_protected_structure", "is_aca",
            "raw_text_length", "extraction_confidence",
        ]
        for key in expected_keys:
            assert key in fields, f"Missing key: {key}"
