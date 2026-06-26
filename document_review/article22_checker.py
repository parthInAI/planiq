"""
PlanIQ — Article 22 Compliance Checker
=======================================
Cross-checks an extracted planning application against the requirements
of Article 22 of the Planning and Development Regulations 2001 (as amended).

Article 22 sets out what a valid planning application must contain.
If any required item is missing or incorrect, the planning authority
can invalidate the application before it is even assessed.

This module:
  1. Takes extracted fields from a planning application PDF
  2. Checks each field against Article 22 requirements
  3. Returns a structured gap report with status per item

Status values:
  PASS    — requirement clearly met
  FAIL    — requirement clearly not met (application will be invalidated)
  WARNING — requirement may not be met (needs human review)
  MISSING — item not found in the document
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional


class CheckStatus(str, Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"
    MISSING = "missing"


@dataclass
class CheckResult:
    """Result of a single Article 22 compliance check."""
    article:     str           # e.g. "Article 22(2)(a)"
    item:        str           # Human-readable name
    status:      CheckStatus
    finding:     str           # What was found (or not found)
    requirement: str           # What Article 22 requires
    action:      str           # What the applicant must do to fix it
    severity:    str           # "critical" | "major" | "minor"


@dataclass
class GapReport:
    """Complete Article 22 gap report for a planning application."""
    application_description: str
    council:                 str
    checked_at:              str
    total_checks:            int
    passed:                  int
    failed:                  int
    warnings:                int
    missing:                 int
    overall_status:          str   # "valid" | "likely_invalid" | "review_required"
    checks:                  list[CheckResult] = field(default_factory=list)
    disclaimer:              str = (
        "This gap report is produced by PlanIQ for guidance only and does not "
        "constitute professional planning advice. Always verify with a registered "
        "planning consultant or your local planning authority before submitting "
        "a planning application."
    )


# ── Article 22 requirement definitions ────────────────────────────────────────

ARTICLE_22_REQUIREMENTS = {

    "22_2_a_form": {
        "article":     "Article 22(2)(a)",
        "item":        "Application form (Form No. 1)",
        "requirement": "Completed Form No. 1 must be submitted. Must include applicant name, "
                       "address, contact details, description of development, type of "
                       "permission sought, and all required signatures.",
        "severity":    "critical",
    },

    "22_2_a_signature": {
        "article":     "Article 22(2)(a)",
        "item":        "Applicant signature on Form No. 1",
        "requirement": "Form No. 1 must be signed by the applicant or their agent. "
                       "An unsigned application form will be invalidated.",
        "severity":    "critical",
    },

    "22_2_b_os_map": {
        "article":     "Article 22(2)(b)",
        "item":        "Ordnance Survey location map",
        "requirement": "An Ordnance Survey map at a scale of not less than 1:1,000 for urban "
                       "sites or 1:2,500 for rural sites. The application site must be "
                       "outlined in red. A north point must be shown.",
        "severity":    "critical",
    },

    "22_2_c_site_layout": {
        "article":     "Article 22(2)(c) and Article 23",
        "item":        "Site layout plan at 1:500 scale",
        "requirement": "Site layout plan at a scale of not less than 1:500. Must show site "
                       "boundary, access points, car parking, drainage, existing and proposed "
                       "structures, and site levels relative to Ordnance Survey datum. "
                       "The same scale must be used throughout the entire drawing.",
        "severity":    "critical",
    },

    "22_2_d_floor_plans": {
        "article":     "Article 22(2)(d) and Article 23",
        "item":        "Floor plans at 1:200 scale",
        "requirement": "Floor plans, elevations, and sections at a scale of not less than "
                       "1:200. For extensions or alterations, existing and proposed works "
                       "must be distinguished by colour — red for proposed, grey or yellow "
                       "for demolition, existing shown in black.",
        "severity":    "critical",
    },

    "22_2_d_elevations": {
        "article":     "Article 22(2)(d)",
        "item":        "Elevations showing adjoining structures",
        "requirement": "Elevations must show the main features of buildings which adjoin "
                       "the proposed structure or any buildings within the curtilage which "
                       "would be materially affected by the proposed development, at a scale "
                       "of not less than 1:200.",
        "severity":    "major",
    },

    "22_2_d_scale_consistency": {
        "article":     "Article 22(2)(d) — Scale consistency",
        "item":        "Consistent scale across all drawings",
        "requirement": "The same scale must be used for the entirety of any individual map "
                       "or drawing submitted in accordance with Article 22. Mixing scales "
                       "on a single drawing is grounds for invalidation.",
        "severity":    "critical",
    },

    "22_2_e_written_statement": {
        "article":     "Article 22(2)(e)",
        "item":        "Written statement describing proposed development",
        "requirement": "A written statement describing the nature and extent of the proposed "
                       "development. For residential developments of 10 or more units, must "
                       "address Part V social and affordable housing requirements.",
        "severity":    "critical",
    },

    "22_2_f_newspaper_notice": {
        "article":     "Article 22(2)(f) and Article 18",
        "item":        "Newspaper notice",
        "requirement": "A copy or original of the newspaper notice must be submitted with "
                       "the application. The application must be lodged within two weeks of "
                       "the publication date of the newspaper notice. The notice must be in "
                       "an approved newspaper, contain the applicant name, planning authority "
                       "name, and description of the development.",
        "severity":    "critical",
    },

    "22_newspaper_timing": {
        "article":     "Article 18(1)(b) — Newspaper notice timing",
        "item":        "Application lodged within 2 weeks of newspaper notice",
        "requirement": "The planning application must be lodged with the planning authority "
                       "within two weeks of the date of publication of the newspaper notice. "
                       "If this deadline is missed the application is invalid.",
        "severity":    "critical",
    },

    "19_site_notice": {
        "article":     "Article 19 — Site notice",
        "item":        "Site notice erected and submitted",
        "requirement": "A site notice must be erected at the site before the application "
                       "is submitted and must remain in place for five weeks. A copy of "
                       "the site notice must be submitted with the application. If the "
                       "notice is removed or becomes illegible it must be replaced immediately.",
        "severity":    "critical",
    },

    "22_2_g_legal_interest": {
        "article":     "Article 22(2)(g)",
        "item":        "Evidence of legal interest in the land",
        "requirement": "If the applicant owns the land — proof of ownership. If the "
                       "applicant is not the owner — written consent of the landowner "
                       "must be submitted with the application.",
        "severity":    "critical",
    },

    "22_2_h_fee": {
        "article":     "Article 22(2)(h) and Articles 156-172",
        "item":        "Appropriate planning fee",
        "requirement": "The correct fee must be submitted with the application, calculated "
                       "in accordance with Articles 156-172 and Schedule 9 of PDR 2001. "
                       "The fee is based on development type and gross floor area. "
                       "An incorrect fee will invalidate the application.",
        "severity":    "critical",
    },

    "22a_protected_structure": {
        "article":     "Article 22A — Protected structure additional requirements",
        "item":        "Photographs and heritage particulars (protected structures only)",
        "requirement": "For development affecting a protected structure or proposed protected "
                       "structure, or the exterior of a structure in an Architectural "
                       "Conservation Area — photographs, plans, and other particulars "
                       "showing how the development would affect the character of the "
                       "structure must be submitted in addition to Article 22 requirements.",
        "severity":    "critical",
    },

}


# ── Compliance checker ────────────────────────────────────────────────────────

class Article22Checker:
    """
    Checks a planning application's extracted fields against Article 22.

    Usage:
        checker = Article22Checker()
        report  = checker.check(extracted_fields)
    """

    def check(self, fields: dict) -> GapReport:
        """
        Run all Article 22 checks against extracted application fields.

        Args:
            fields: dict from PDFFieldExtractor.extract()

        Returns:
            GapReport with status per check item
        """
        results = []

        results.append(self._check_form(fields))
        results.append(self._check_signature(fields))
        results.append(self._check_os_map(fields))
        results.append(self._check_site_layout(fields))
        results.append(self._check_floor_plans(fields))
        results.append(self._check_elevations(fields))
        results.append(self._check_scale_consistency(fields))
        results.append(self._check_written_statement(fields))
        results.append(self._check_newspaper_notice(fields))
        results.append(self._check_newspaper_timing(fields))
        results.append(self._check_site_notice(fields))
        results.append(self._check_legal_interest(fields))
        results.append(self._check_fee(fields))

        if fields.get("is_protected_structure") or fields.get("is_aca"):
            results.append(self._check_protected_structure(fields))

        passed   = sum(1 for r in results if r.status == CheckStatus.PASS)
        failed   = sum(1 for r in results if r.status == CheckStatus.FAIL)
        warnings = sum(1 for r in results if r.status == CheckStatus.WARNING)
        missing  = sum(1 for r in results if r.status == CheckStatus.MISSING)

        if failed > 0:
            overall = "likely_invalid"
        elif warnings > 2 or missing > 2:
            overall = "review_required"
        else:
            overall = "valid"

        from datetime import datetime
        return GapReport(
            application_description = fields.get("development_description", "Unknown"),
            council                 = fields.get("planning_authority", "Unknown"),
            checked_at              = datetime.utcnow().isoformat(),
            total_checks            = len(results),
            passed                  = passed,
            failed                  = failed,
            warnings                = warnings,
            missing                 = missing,
            overall_status          = overall,
            checks                  = results,
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_form(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_a_form"]
        has_form = f.get("has_application_form", False)
        has_name = bool(f.get("applicant_name", "").strip())
        has_desc = bool(f.get("development_description", "").strip())

        if has_form and has_name and has_desc:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Application form detected with applicant name and development description.",
                action="No action required.")
        elif has_form and (not has_name or not has_desc):
            return CheckResult(**req,
                status=CheckStatus.WARNING,
                finding=f"Form detected but {'applicant name' if not has_name else 'development description'} appears missing.",
                action="Verify that all required fields on Form No. 1 are completed before submission.")
        else:
            return CheckResult(**req,
                status=CheckStatus.MISSING,
                finding="Application form (Form No. 1) not clearly identified in the document.",
                action="Ensure a completed Form No. 1 is included as the first document in your application.")

    def _check_signature(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_a_signature"]
        signed = f.get("form_signed", None)

        if signed is True:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Signature detected on application form.",
                action="No action required.")
        elif signed is False:
            return CheckResult(**req,
                status=CheckStatus.FAIL,
                finding="No signature detected on the application form.",
                action="The applicant or their agent must sign Form No. 1 before submission. "
                       "An unsigned form will result in immediate invalidation.")
        else:
            return CheckResult(**req,
                status=CheckStatus.WARNING,
                finding="Could not confirm whether the application form is signed.",
                action="Manually verify that Form No. 1 is signed before submission.")

    def _check_os_map(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_b_os_map"]
        has_map    = f.get("has_os_map", False)
        has_red    = f.get("site_outlined_red", None)
        map_scale  = f.get("os_map_scale", "")

        if has_map:
            issues = []
            if has_red is False:
                issues.append("site boundary not outlined in red")
            if map_scale and not self._scale_ok(map_scale, min_urban=1000, min_rural=2500):
                issues.append(f"scale {map_scale} may be too small — 1:1,000 urban or 1:2,500 rural required")

            if not issues:
                return CheckResult(**req,
                    status=CheckStatus.PASS,
                    finding=f"Ordnance Survey map detected{f' at scale {map_scale}' if map_scale else ''}.",
                    action="No action required.")
            else:
                return CheckResult(**req,
                    status=CheckStatus.WARNING,
                    finding=f"OS map detected but: {', '.join(issues)}.",
                    action="Correct the identified issues before submission.")
        else:
            return CheckResult(**req,
                status=CheckStatus.MISSING,
                finding="No Ordnance Survey location map detected in the application.",
                action="Include an Ordnance Survey map at 1:1,000 (urban) or 1:2,500 (rural) scale "
                       "with the site outlined in red and a north point shown.")

    def _check_site_layout(self, f: dict) -> CheckResult:
        req   = ARTICLE_22_REQUIREMENTS["22_2_c_site_layout"]
        has   = f.get("has_site_layout_plan", False)
        scale = f.get("site_layout_scale", "")

        if has:
            if scale and not self._scale_ok(scale, min_urban=500):
                return CheckResult(**req,
                    status=CheckStatus.FAIL,
                    finding=f"Site layout plan detected but scale {scale} is less than the required 1:500.",
                    action="Redraw the site layout plan at a scale of not less than 1:500.")
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding=f"Site layout plan detected{f' at scale {scale}' if scale else ''}.",
                action="No action required.")
        return CheckResult(**req,
            status=CheckStatus.MISSING,
            finding="No site layout plan detected.",
            action="Include a site layout plan at a scale of not less than 1:500 showing "
                   "site boundary, access, parking, drainage, and existing and proposed structures.")

    def _check_floor_plans(self, f: dict) -> CheckResult:
        req   = ARTICLE_22_REQUIREMENTS["22_2_d_floor_plans"]
        has   = f.get("has_floor_plans", False)
        scale = f.get("floor_plan_scale", "")
        coloured = f.get("drawings_coloured", None)
        is_extension = f.get("is_extension_or_alteration", False)

        if has:
            issues = []
            if scale and not self._scale_ok(scale, min_urban=200):
                issues.append(f"scale {scale} is less than required 1:200")
            if is_extension and coloured is False:
                issues.append("existing and proposed works must be distinguished by colour")

            if not issues:
                return CheckResult(**req,
                    status=CheckStatus.PASS,
                    finding=f"Floor plans detected{f' at scale {scale}' if scale else ''}.",
                    action="No action required.")
            else:
                return CheckResult(**req,
                    status=CheckStatus.FAIL,
                    finding=f"Floor plans detected but: {', '.join(issues)}.",
                    action="Correct the identified issues. Redraw plans at 1:200 minimum. "
                           "For extensions — use red for proposed works, existing shown in black.")
        return CheckResult(**req,
            status=CheckStatus.MISSING,
            finding="No floor plans detected.",
            action="Include floor plans, elevations, and sections at a scale of not less than 1:200.")

    def _check_elevations(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_d_elevations"]
        has = f.get("has_elevations", False)

        if has:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Elevations detected.",
                action="Verify that elevations show all adjoining structures at 1:200 scale.")
        return CheckResult(**req,
            status=CheckStatus.WARNING,
            finding="Elevations not clearly identified.",
            action="Include elevations showing all sides of the proposed structure and "
                   "main features of any adjoining buildings at 1:200 scale.")

    def _check_scale_consistency(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_d_scale_consistency"]
        consistent = f.get("scales_consistent", None)

        if consistent is True:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Scales appear consistent across drawings.",
                action="No action required.")
        elif consistent is False:
            return CheckResult(**req,
                status=CheckStatus.FAIL,
                finding="Inconsistent scales detected across drawings.",
                action="Redraw all plans using the same scale throughout each individual drawing. "
                       "Article 22 requires the same scale to be used for the entirety of any drawing.")
        return CheckResult(**req,
            status=CheckStatus.WARNING,
            finding="Could not verify scale consistency across all drawings.",
            action="Manually verify that each drawing uses a consistent scale throughout.")

    def _check_written_statement(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_e_written_statement"]
        has  = f.get("has_written_statement", False)
        desc = f.get("development_description", "")

        if has or len(desc) > 100:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Written statement or development description detected.",
                action="Verify that the description matches the drawings submitted.")
        return CheckResult(**req,
            status=CheckStatus.MISSING,
            finding="No written statement describing the proposed development detected.",
            action="Include a written statement describing the nature and extent of the "
                   "proposed development.")

    def _check_newspaper_notice(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_f_newspaper_notice"]
        has = f.get("has_newspaper_notice", False)
        newspaper = f.get("newspaper_name", "")

        if has:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding=f"Newspaper notice detected{f' in {newspaper}' if newspaper else ''}.",
                action="Verify the notice appeared in a council-approved newspaper.")
        return CheckResult(**req,
            status=CheckStatus.MISSING,
            finding="No newspaper notice detected in the application documents.",
            action="A newspaper notice must be published in an approved newspaper before submission. "
                   "Include a copy or original of the notice with your application.")

    def _check_newspaper_timing(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_newspaper_timing"]
        pub_date  = f.get("newspaper_publication_date")
        app_date  = f.get("application_lodgement_date")

        if pub_date and app_date:
            try:
                if isinstance(pub_date, str):
                    pub_date = date.fromisoformat(pub_date)
                if isinstance(app_date, str):
                    app_date = date.fromisoformat(app_date)

                days_diff = (app_date - pub_date).days
                if days_diff < 0:
                    return CheckResult(**req,
                        status=CheckStatus.FAIL,
                        finding=f"Application date ({app_date}) is before newspaper publication date ({pub_date}).",
                        action="The application cannot be lodged before the newspaper notice is published.")
                elif days_diff <= 14:
                    return CheckResult(**req,
                        status=CheckStatus.PASS,
                        finding=f"Application lodged {days_diff} days after newspaper notice — within the 14-day window.",
                        action="No action required.")
                else:
                    return CheckResult(**req,
                        status=CheckStatus.FAIL,
                        finding=f"Application lodged {days_diff} days after newspaper notice — exceeds the 14-day limit.",
                        action="The application is invalid. A new newspaper notice must be published "
                               "and the application re-lodged within 2 weeks of that notice.")
            except Exception:
                pass

        return CheckResult(**req,
            status=CheckStatus.WARNING,
            finding="Could not verify newspaper notice timing — dates not clearly extracted.",
            action="Manually verify that the application is lodged within 2 weeks of the "
                   "newspaper notice publication date.")

    def _check_site_notice(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["19_site_notice"]
        has = f.get("has_site_notice", False)

        if has:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Site notice copy detected in application documents.",
                action="Ensure the site notice is still legible and in place at the site.")
        return CheckResult(**req,
            status=CheckStatus.MISSING,
            finding="No site notice copy detected in the application documents.",
            action="A site notice must be erected at the site before submission. Include a copy "
                   "of the site notice with the application. The notice must remain in place "
                   "for 5 weeks from the date of receipt by the planning authority.")

    def _check_legal_interest(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_g_legal_interest"]
        is_owner   = f.get("applicant_is_owner", None)
        has_consent = f.get("has_landowner_consent", False)

        if is_owner is True:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Applicant appears to be the landowner.",
                action="Ensure proof of ownership is included if requested by the planning authority.")
        elif is_owner is False and has_consent:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Landowner consent letter detected.",
                action="No action required.")
        elif is_owner is False and not has_consent:
            return CheckResult(**req,
                status=CheckStatus.FAIL,
                finding="Applicant does not appear to be the landowner and no landowner consent detected.",
                action="Include written consent from the landowner with the application. "
                       "Without this the application will be invalidated.")
        return CheckResult(**req,
            status=CheckStatus.WARNING,
            finding="Could not determine applicant's legal interest in the land.",
            action="Confirm whether the applicant owns the land. If not, include written "
                   "landowner consent with the application.")

    def _check_fee(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22_2_h_fee"]
        has_fee     = f.get("has_fee_included", None)
        fee_amount  = f.get("fee_amount", None)
        floor_area  = f.get("gross_floor_area_sqm", None)

        if has_fee is True:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding=f"Fee detected{f': EUR {fee_amount:.2f}' if fee_amount else ''}.",
                action="Verify that the fee is correctly calculated for the development type and floor area.")
        elif has_fee is False:
            return CheckResult(**req,
                status=CheckStatus.FAIL,
                finding="No planning fee detected in the application.",
                action="Calculate and include the correct fee per Articles 156-172 and Schedule 9 "
                       "of the Planning and Development Regulations 2001.")
        return CheckResult(**req,
            status=CheckStatus.WARNING,
            finding="Could not confirm whether the correct fee has been included.",
            action="Verify that the planning fee is included and correctly calculated for "
                   "the development type and gross floor area.")

    def _check_protected_structure(self, f: dict) -> CheckResult:
        req = ARTICLE_22_REQUIREMENTS["22a_protected_structure"]
        has_photos   = f.get("has_heritage_photographs", False)
        has_heritage = f.get("has_heritage_particulars", False)

        if has_photos and has_heritage:
            return CheckResult(**req,
                status=CheckStatus.PASS,
                finding="Heritage photographs and particulars detected.",
                action="Verify that the photographs clearly show how the development affects "
                       "the character of the protected structure.")
        elif has_photos or has_heritage:
            return CheckResult(**req,
                status=CheckStatus.WARNING,
                finding="Some heritage documentation detected but may be incomplete.",
                action="Ensure both photographs and written heritage particulars are included "
                       "showing how the development affects the character of the structure.")
        return CheckResult(**req,
            status=CheckStatus.FAIL,
            finding="No heritage photographs or particulars detected for a protected structure "
                    "or ACA application.",
            action="Article 22A requires photographs, plans, and particulars showing how the "
                   "development would affect the character of the protected structure. These "
                   "must be submitted in addition to the standard Article 22 requirements.")

    # ── Scale validation helper ───────────────────────────────────────────────

    def _scale_ok(self, scale_str: str, min_urban: int = 500, min_rural: int = None) -> bool:
        """
        Check if a scale string like '1:200' or '1:500' meets the minimum.
        Returns True if the scale is equal to or larger than min_urban.
        (Smaller denominator = larger scale = more detail = better)
        """
        match = re.search(r'1[:\s/](\d+)', str(scale_str).replace(',', ''))
        if not match:
            return True  # Cannot parse — do not flag
        denominator = int(match.group(1))
        return denominator <= min_urban
