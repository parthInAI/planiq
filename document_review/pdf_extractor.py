"""
PlanIQ — Planning Application PDF Field Extractor
==================================================
Extracts structured fields from a planning application PDF upload.

Looks for:
  - Application form fields (applicant name, description, signatures)
  - Drawing references and scales
  - Newspaper notice details and dates
  - Site notice copy
  - Fee information
  - Legal interest declarations
  - Protected structure indicators

Returns a dict of extracted fields consumed by Article22Checker.
"""

from __future__ import annotations

import re
import io
from pathlib import Path
from typing import Optional
from datetime import date, datetime


class PDFFieldExtractor:
    """
    Extracts planning application fields from a PDF file.

    Works with:
      - Completed Form No. 1 PDFs
      - Scanned applications (with basic text detection)
      - Digital planning applications submitted via NOPPS
    """

    # ── Scale patterns ────────────────────────────────────────────────────────
    SCALE_PATTERN = re.compile(
        r'1\s*[:/]\s*(\d[\d,]*)',
        re.IGNORECASE
    )

    # ── Date patterns ─────────────────────────────────────────────────────────
    DATE_PATTERNS = [
        re.compile(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})'),
        re.compile(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
                   r'September|October|November|December)\s+(\d{4})', re.IGNORECASE),
        re.compile(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})'),
    ]

    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }

    def extract(self, pdf_bytes: bytes) -> dict:
        """
        Extract fields from planning application PDF bytes.

        Args:
            pdf_bytes: Raw bytes of the uploaded PDF

        Returns:
            dict of extracted fields for Article22Checker
        """
        text = self._extract_text(pdf_bytes)
        text_lower = text.lower()

        fields = {
            # ── Form fields ─────────────────────────────────────────────
            "has_application_form":       self._detect_form(text),
            "applicant_name":             self._extract_applicant_name(text),
            "planning_authority":         self._extract_planning_authority(text),
            "development_description":    self._extract_description(text),
            "permission_type":            self._extract_permission_type(text_lower),
            "form_signed":                self._detect_signature(text),

            # ── Maps and drawings ────────────────────────────────────────
            "has_os_map":                 self._detect_os_map(text_lower),
            "site_outlined_red":          self._detect_red_outline(text_lower),
            "os_map_scale":               self._extract_os_map_scale(text),
            "has_site_layout_plan":       self._detect_site_layout(text_lower),
            "site_layout_scale":          self._extract_site_layout_scale(text),
            "has_floor_plans":            self._detect_floor_plans(text_lower),
            "floor_plan_scale":           self._extract_floor_plan_scale(text),
            "has_elevations":             self._detect_elevations(text_lower),
            "has_sections":               self._detect_sections(text_lower),
            "drawings_coloured":          self._detect_colouring(text_lower),
            "scales_consistent":          self._check_scale_consistency(text),
            "is_extension_or_alteration": self._detect_extension(text_lower),

            # ── Notices ──────────────────────────────────────────────────
            "has_newspaper_notice":       self._detect_newspaper_notice(text_lower),
            "newspaper_name":             self._extract_newspaper_name(text),
            "newspaper_publication_date": self._extract_newspaper_date(text),
            "has_site_notice":            self._detect_site_notice(text_lower),
            "application_lodgement_date": self._extract_lodgement_date(text),

            # ── Legal interest ───────────────────────────────────────────
            "applicant_is_owner":         self._detect_owner(text_lower),
            "has_landowner_consent":      self._detect_landowner_consent(text_lower),

            # ── Fee ──────────────────────────────────────────────────────
            "has_fee_included":           self._detect_fee(text_lower),
            "fee_amount":                 self._extract_fee_amount(text),
            "gross_floor_area_sqm":       self._extract_floor_area(text),

            # ── Protected structure ──────────────────────────────────────
            "is_protected_structure":     self._detect_protected_structure(text_lower),
            "is_aca":                     self._detect_aca(text_lower),
            "has_heritage_photographs":   self._detect_heritage_photos(text_lower),
            "has_heritage_particulars":   self._detect_heritage_particulars(text_lower),

            # ── Meta ─────────────────────────────────────────────────────
            "raw_text_length":            len(text),
            "extraction_confidence":      self._score_confidence(text),
        }

        return fields

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages  = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            return ""

    # ── Form detection ────────────────────────────────────────────────────────

    def _detect_form(self, text: str) -> bool:
        indicators = [
            "form no", "form 1", "planning application",
            "applicant", "planning authority", "nature of application",
        ]
        score = sum(1 for i in indicators if i in text.lower())
        return score >= 3

    def _extract_applicant_name(self, text: str) -> str:
        patterns = [
            r'applicant[:\s]+([A-Z][a-zA-Z\s\-\']+?)(?:\n|,|\.)',
            r'name of applicant[:\s]+([A-Z][a-zA-Z\s\-\']+?)(?:\n|,|\.)',
            r'i/we[,\s]+([A-Z][a-zA-Z\s\-\']+?)[,\s]+of',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if 5 <= len(name) <= 80:
                    return name
        return ""

    def _extract_planning_authority(self, text: str) -> str:
        councils = [
            "Dublin City Council", "Fingal County Council",
            "South Dublin County Council", "Dun Laoghaire-Rathdown",
            "Cork City Council", "Cork County Council",
            "Galway City Council", "Galway County Council",
            "Limerick City and County Council", "Waterford City and County Council",
            "Kerry County Council", "Kildare County Council",
            "Meath County Council", "Wicklow County Council",
            "Wexford County Council", "Kilkenny County Council",
            "Tipperary County Council", "Laois County Council",
            "Longford County Council", "Louth County Council",
            "Monaghan County Council", "Roscommon County Council",
            "Clare County Council", "Carlow County Council",
        ]
        text_lower = text.lower()
        for council in councils:
            if council.lower() in text_lower:
                return council
        patterns = [
            r'planning authority[:\s]+([A-Z][a-zA-Z\s]+Council)',
            r'to[:\s]+([A-Z][a-zA-Z\s]+Council)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_description(self, text: str) -> str:
        patterns = [
            r'description of proposed development[:\s]+(.{20,500}?)(?:\n\n|\Z)',
            r'nature of development[:\s]+(.{20,300}?)(?:\n\n|\Z)',
            r'proposed development[:\s]+(.{20,300}?)(?:\n\n|\Z)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                desc = match.group(1).strip()
                desc = re.sub(r'\s+', ' ', desc)
                if len(desc) > 20:
                    return desc[:500]
        return ""

    def _extract_permission_type(self, text: str) -> str:
        if "retention" in text:
            return "retention"
        if "outline" in text:
            return "outline"
        if "extension of duration" in text:
            return "extension_of_duration"
        if "permission" in text:
            return "permission"
        return "unknown"

    def _detect_signature(self, text: str) -> Optional[bool]:
        text_lower = text.lower()
        signed_indicators = [
            "signed:", "signature:", "/s/", "dated and signed",
            "applicant's signature", "agent's signature",
        ]
        unsigned_indicators = [
            "please sign here", "sign here", "[signature]", "____",
        ]
        has_signed   = any(i in text_lower for i in signed_indicators)
        has_unsigned = any(i in text_lower for i in unsigned_indicators)

        if has_signed and not has_unsigned:
            return True
        if has_unsigned and not has_signed:
            return False
        return None

    # ── Maps and drawings ─────────────────────────────────────────────────────

    def _detect_os_map(self, text: str) -> bool:
        indicators = [
            "ordnance survey", "os map", "location map",
            "site location", "1:1,000", "1:2,500", "1:1000", "1:2500",
            "red line boundary", "site boundary",
        ]
        return any(i in text for i in indicators)

    def _detect_red_outline(self, text: str) -> Optional[bool]:
        if any(i in text for i in ["outlined in red", "red line", "red boundary", "site edged red"]):
            return True
        if any(i in text for i in ["outlined in blue", "blue boundary"]):
            return False
        return None

    def _extract_os_map_scale(self, text: str) -> str:
        context_patterns = [
            r'(?:location|os|ordnance|site location)[^\n]*?1[:/](\d[\d,]*)',
            r'1[:/](1[,\s]?000|2[,\s]?500)',
        ]
        for pattern in context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"1:{match.group(1).replace(',', '').replace(' ', '')}"
        return ""

    def _detect_site_layout(self, text: str) -> bool:
        return any(i in text for i in [
            "site layout", "site plan", "layout plan",
            "block plan", "1:500", "1/500",
        ])

    def _extract_site_layout_scale(self, text: str) -> str:
        match = re.search(
            r'(?:site layout|site plan|layout plan|block plan)[^\n]*?1[:/](\d[\d,]*)',
            text, re.IGNORECASE
        )
        if match:
            return f"1:{match.group(1).replace(',', '')}"
        match = re.search(r'1[:/](500|250|200)', text, re.IGNORECASE)
        if match:
            return f"1:{match.group(1)}"
        return ""

    def _detect_floor_plans(self, text: str) -> bool:
        return any(i in text for i in [
            "floor plan", "ground floor", "first floor", "second floor",
            "roof plan", "plan view", "1:100", "1:200", "1/200",
        ])

    def _extract_floor_plan_scale(self, text: str) -> str:
        match = re.search(
            r'(?:floor plan|ground floor|first floor|elevation)[^\n]*?1[:/](\d[\d,]*)',
            text, re.IGNORECASE
        )
        if match:
            return f"1:{match.group(1).replace(',', '')}"
        for scale in ["1:50", "1:100", "1:200"]:
            if scale in text or scale.replace(":", "/") in text:
                return scale
        return ""

    def _detect_elevations(self, text: str) -> bool:
        return any(i in text for i in [
            "elevation", "north elevation", "south elevation",
            "east elevation", "west elevation", "front elevation", "rear elevation",
        ])

    def _detect_sections(self, text: str) -> bool:
        return any(i in text for i in [
            "section a-a", "section b-b", "cross section",
            "longitudinal section", "section through",
        ])

    def _detect_colouring(self, text: str) -> Optional[bool]:
        coloured = any(i in text for i in [
            "shown in red", "coloured red", "existing shown",
            "proposed works in red", "red for new", "hatched",
        ])
        not_coloured = "uncoloured" in text
        if coloured:
            return True
        if not_coloured:
            return False
        return None

    def _check_scale_consistency(self, text: str) -> Optional[bool]:
        scales = self.SCALE_PATTERN.findall(text.replace(',', ''))
        if not scales:
            return None
        scale_nums = [int(s) for s in scales if s.isdigit()]
        if not scale_nums:
            return None
        unique = set(scale_nums)
        if len(unique) <= 3:
            return True
        if max(unique) / min(unique) > 10:
            return False
        return None

    def _detect_extension(self, text: str) -> bool:
        return any(i in text for i in [
            "extension", "alteration", "renovation", "conversion",
            "refurbishment", "addition to", "proposed addition",
        ])

    # ── Notices ───────────────────────────────────────────────────────────────

    def _detect_newspaper_notice(self, text: str) -> bool:
        return any(i in text for i in [
            "newspaper notice", "public notice", "notice of intention",
            "notice published", "irish times", "irish examiner",
            "irish independent", "indo", "evening echo",
        ])

    def _extract_newspaper_name(self, text: str) -> str:
        newspapers = [
            "The Irish Times", "Irish Times",
            "Irish Independent", "The Independent",
            "Irish Examiner", "The Examiner",
            "Evening Echo", "Evening Herald",
        ]
        for paper in newspapers:
            if paper.lower() in text.lower():
                return paper
        return ""

    def _extract_newspaper_date(self, text: str) -> Optional[str]:
        context = re.search(
            r'(?:published|appeared|newspaper notice)[^\n]{0,100}?'
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            text, re.IGNORECASE
        )
        if context:
            return self._normalise_date(context.group(1))
        return None

    def _extract_lodgement_date(self, text: str) -> Optional[str]:
        context = re.search(
            r'(?:date of application|lodged on|submitted on|date lodged)[^\n]{0,50}?'
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            text, re.IGNORECASE
        )
        if context:
            return self._normalise_date(context.group(1))
        return None

    def _normalise_date(self, date_str: str) -> Optional[str]:
        try:
            parts = re.split(r'[/\-\.]', date_str)
            if len(parts) == 3:
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                if y < 100:
                    y += 2000
                return date(y, m, d).isoformat()
        except Exception:
            pass
        return None

    def _detect_site_notice(self, text: str) -> bool:
        return any(i in text for i in [
            "site notice", "notice erected", "planning notice",
            "notice placed", "public notice at site",
        ])

    # ── Legal interest ────────────────────────────────────────────────────────

    def _detect_owner(self, text: str) -> Optional[bool]:
        owner_indicators = [
            "i am the owner", "we are the owner", "owner of the", "as owner",
            "freehold owner", "fee simple",
        ]
        non_owner_indicators = [
            "consent of the owner", "landowner consent", "owner's permission",
            "with the consent", "agent acting on behalf",
        ]
        is_owner     = any(i in text for i in owner_indicators)
        not_owner    = any(i in text for i in non_owner_indicators)

        if is_owner and not not_owner:
            return True
        if not_owner and not is_owner:
            return False
        return None

    def _detect_landowner_consent(self, text: str) -> bool:
        return any(i in text for i in [
            "landowner consent", "owner's consent", "owner consent",
            "consent of the owner", "written consent", "consent letter",
            "i hereby consent", "we hereby consent",
        ])

    # ── Fee ──────────────────────────────────────────────────────────────────

    def _detect_fee(self, text: str) -> Optional[bool]:
        fee_indicators = [
            "planning fee", "application fee", "fee enclosed",
            "fee paid", "cheque enclosed", "bank draft",
        ]
        has_fee = any(i in text for i in fee_indicators)
        no_fee  = any(i in text for i in ["no fee required", "exempt from fee", "fee exemption"])

        if has_fee:
            return True
        if no_fee:
            return None  # May be legitimately exempt
        return None

    def _extract_fee_amount(self, text: str) -> Optional[float]:
        match = re.search(
            r'(?:fee|amount)[^\n]{0,30}?(?:EUR?|€)\s*([\d,]+(?:\.\d{2})?)',
            text, re.IGNORECASE
        )
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        return None

    def _extract_floor_area(self, text: str) -> Optional[float]:
        match = re.search(
            r'(?:gross floor area|floor area|total area)[^\n]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*m|m2|m²)',
            text, re.IGNORECASE
        )
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        return None

    # ── Protected structure ───────────────────────────────────────────────────

    def _detect_protected_structure(self, text: str) -> bool:
        return any(i in text for i in [
            "protected structure", "rps", "record of protected structures",
            "proposed protected structure",
        ])

    def _detect_aca(self, text: str) -> bool:
        return any(i in text for i in [
            "architectural conservation area", "aca",
        ])

    def _detect_heritage_photos(self, text: str) -> bool:
        return any(i in text for i in [
            "photograph", "photo", "existing elevation", "existing view",
            "heritage photograph", "survey photograph",
        ])

    def _detect_heritage_particulars(self, text: str) -> bool:
        return any(i in text for i in [
            "heritage statement", "conservation report", "architectural heritage",
            "impact on character", "affect the character",
            "protected structure report",
        ])

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _score_confidence(self, text: str) -> str:
        """
        Score extraction confidence based on how much text was found
        and how many planning-specific terms were detected.
        """
        if len(text) < 500:
            return "low"
        planning_terms = [
            "planning", "permission", "development", "application",
            "council", "applicant", "scale", "elevation", "site",
        ]
        score = sum(1 for t in planning_terms if t in text.lower())
        if score >= 7:
            return "high"
        if score >= 4:
            return "medium"
        return "low"
