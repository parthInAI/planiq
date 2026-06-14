"""
PlanIQ — Document Scraper
Fetches Irish planning legislation and council documents from live sources.

Sources ingested in Step 1 (MVP — Dublin City + Fingal):
  - irishstatutebook.ie  (PDA 2000, PDA 2024, PDR 2001 Schedule 2)
  - gov.ie               (DHLGH ministerial guidelines, NPF)
  - dublincity.ie        (Dublin City Council development plan)
  - fingal.ie            (Fingal County Council development plan)

Each scrape produces: raw text + source metadata → fed to SemanticChunker.
Rate limiting and retry logic built in — we are a responsible scraper.
"""

import time
import hashlib
import requests
from datetime import date
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ── Constants ───────────────────────────────
HEADERS = {
    "User-Agent": (
        "PlanIQ/1.0 (AI planning guidance tool; research use; "
        "contact: parthpandya474@gmail.com)"
    )
}
REQUEST_TIMEOUT   = 30   # seconds
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 2    # seconds between retries
RATE_LIMIT_DELAY  = 1.5  # seconds between requests (be a good citizen)
RAW_DATA_DIR      = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Source registry ──────────────────────────
# Each entry = one document to ingest
# Extend this list as we add more councils / documents
PLANNING_SOURCES = [

    # ── National primary legislation ──────────
    {
        "id":            "pda_2000_overview",
        "title":         "Planning and Development Act 2000 — Overview",
        "url":           "https://www.irishstatutebook.ie/eli/2000/act/30/enacted/en/html",
        "document_type": "primary_act",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2000,
        "effective_date": date(2000, 11, 28),
        "confidence":    "high",
        "is_verbatim":   True,
    },
    {
        "id":            "pda_2024_overview",
        "title":         "Planning and Development Act 2024",
        "url":           "https://www.irishstatutebook.ie/eli/2024/act/34/enacted/en/html",
        "document_type": "primary_act",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2024,
        "effective_date": date(2024, 10, 17),
        "confidence":    "high",
        "is_verbatim":   True,
    },

    # ── Exempted development (Schedule 2 PDR 2001) ──
    {
        "id":            "pdr_2001_schedule2",
        "title":         "Planning and Development Regulations 2001 — Schedule 2 (Exempted Development Classes)",
        "url":           "https://www.irishstatutebook.ie/eli/2001/si/600/made/en/print",
        "document_type": "exemption_schedule",
        "jurisdiction":  "national",
        "si_number":     "S.I. No. 600 of 2001",
        "act_year":      2001,
        "effective_date": date(2001, 10, 1),
        "confidence":    "high",
        "is_verbatim":   True,
    },
    {
        "id":            "exemption_solar_2022",
        "title":         "Planning and Development (Exempted Development) (No. 3) Regulations 2022 — Class 20F Solar",
        "url":           "https://www.irishstatutebook.ie/eli/2022/si/493/made/en/print",
        "document_type": "secondary_si",
        "jurisdiction":  "national",
        "si_number":     "S.I. No. 493 of 2022",
        "act_year":      2022,
        "effective_date": date(2022, 9, 15),
        "confidence":    "high",
        "is_verbatim":   True,
    },

    # ── National policy ───────────────────────
    {
        "id":            "npf_2040",
        "title":         "National Planning Framework — Project Ireland 2040",
        "url":           "https://www.npf.ie/wp-content/uploads/Project-Ireland-2040-NPF.pdf",
        "document_type": "national_policy",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2018,
        "effective_date": date(2018, 2, 16),
        "confidence":    "high",
        "is_verbatim":   False,
    },

    # ── Ministerial guidelines ────────────────
    {
        "id":            "guidelines_rural_housing",
        "title":         "Sustainable Rural Housing Guidelines for Planning Authorities (DHLGH 2005)",
        "url":           "https://www.gov.ie/pdf/?file=https://assets.gov.ie/114242/52001fa8-a7f0-423c-9964-b17c85e05de4.pdf",
        "document_type": "ministerial_guide",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2005,
        "effective_date": date(2005, 4, 1),
        "confidence":    "medium",
        "is_verbatim":   False,
    },

    # ── Seed documents (manually verified statutory text) ──────
    {
        "id":            "schedule2_part1_seed",
        "title":         "Planning and Development Regulations 2001 — Schedule 2 Part 1 Classes 1-7 (Exempted Development)",
        "url":           "local://schedule2_part1_classes1to7_seed.txt",
        "document_type": "exemption_schedule",
        "jurisdiction":  "national",
        "si_number":     "S.I. No. 600 of 2001",
        "act_year":      2001,
        "effective_date": date(2002, 3, 11),
        "confidence":    "high",
        "is_verbatim":   True,
    },

    # ── Local authority development plans (local PDF files) ──
    {
        "id":            "dublin_city_devplan_2022",
        "title":         "Dublin City Development Plan 2022-2028 — Written Statement",
        "url":           "local://dublin_city_devplan_2022.pdf",
        "document_type": "council_devplan",
        "jurisdiction":  "dublin_city",
        "si_number":     "",
        "act_year":      2022,
        "effective_date": date(2022, 12, 14),
        "confidence":    "high",
        "is_verbatim":   True,
    },

    # ── Citizens Information (plain-English reference) ──
    {
        "id":            "citizens_info_planning",
        "title":         "Planning Permission — Citizens Information",
        "url":           "https://www.citizensinformation.ie/en/housing/planning-permission/planning-permission/",
        "document_type": "ministerial_guide",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2024,
        "effective_date": date(2024, 1, 1),
        "confidence":    "medium",
        "is_verbatim":   False,
    },
    {
        "id":            "citizens_info_exempted",
        "title":         "Exempted Development — Citizens Information",
        "url":           "https://www.citizensinformation.ie/en/housing/planning-permission/exempted-development/",
        "document_type": "exemption_schedule",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2024,
        "effective_date": date(2024, 1, 1),
        "confidence":    "medium",
        "is_verbatim":   False,
    },
    {
        "id":            "citizens_info_appeal",
        "title":         "Appealing a Planning Decision — Citizens Information",
        "url":           "https://www.citizensinformation.ie/en/housing/planning-permission/appealing-planning-permission-decision/",
        "document_type": "ministerial_guide",
        "jurisdiction":  "national",
        "si_number":     "",
        "act_year":      2024,
        "effective_date": date(2024, 1, 1),
        "confidence":    "medium",
        "is_verbatim":   False,
    },
]


class PlanningDocumentScraper:
    """
    Fetches planning documents from live Irish government sources.
    Caches raw HTML/text to disk to avoid re-scraping.
    Respects robots.txt via rate limiting and proper User-Agent.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache   = use_cache
        self.session     = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_all(self) -> list[dict]:
        """
        Fetch all sources in PLANNING_SOURCES registry.
        Returns list of dicts: {metadata, raw_text} for the chunker.
        """
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching planning documents...", total=len(PLANNING_SOURCES))

            for source in PLANNING_SOURCES:
                progress.update(task, description=f"Fetching: {source['title'][:50]}...")
                result = self._fetch_one(source)
                if result:
                    results.append(result)
                time.sleep(RATE_LIMIT_DELAY)
                progress.advance(task)

        console.log(f"[green]✓[/] Fetched {len(results)}/{len(PLANNING_SOURCES)} documents")
        return results

    def fetch_source(self, source_id: str) -> Optional[dict]:
        """Fetch a single source by its ID."""
        source = next((s for s in PLANNING_SOURCES if s["id"] == source_id), None)
        if not source:
            console.log(f"[red]✗[/] Source not found: {source_id}")
            return None
        return self._fetch_one(source)

    # ── private ──────────────────────────────

    def _fetch_one(self, source: dict) -> Optional[dict]:
        url       = source["url"]
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = RAW_DATA_DIR / f"{source['id']}_{cache_key[:8]}.txt"

        # ── Local PDF file handling ───────────────
        if url.startswith("local://"):
            filename   = url.replace("local://", "")
            local_path = RAW_DATA_DIR / filename
            if not local_path.exists():
                console.log(f"  [red]✗ Local file not found:[/] {local_path}")
                console.log(f"  [yellow]  Place the PDF at:[/] data/raw/{filename}")
                return None
            console.log(f"  [dim]local file:[/] {filename}")
            # Handle txt files directly
            if local_path.suffix.lower() == ".txt":
                raw_text = local_path.read_text(encoding="utf-8", errors="ignore")
            else:
                raw_text = self._extract_pdf_text(local_path.read_bytes(), str(local_path))
            if not raw_text:
                return None
            return {
                "metadata":   source,
                "raw_text":   raw_text,
                "char_count": len(raw_text),
            }

        # Return cached version if available
        if self.use_cache and cache_path.exists():
            console.log(f"  [dim]cache hit:[/] {source['id']}")
            raw_text = cache_path.read_text(encoding="utf-8")
        else:
            raw_text = self._fetch_with_retry(url)
            if raw_text is None:
                return None
            # Save to cache
            cache_path.write_text(raw_text, encoding="utf-8")

        return {
            "metadata": source,
            "raw_text": raw_text,
            "char_count": len(raw_text),
        }

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """HTTP GET with retry + exponential backoff."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()

                if url.endswith(".pdf"):
                    # PDF: write bytes to temp file and extract text
                    return self._extract_pdf_text(resp.content, url)
                else:
                    return self._extract_html_text(resp.text, url)

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    console.log(f"  [red]404:[/] {url}")
                    return None
                console.log(f"  [yellow]HTTP {resp.status_code}[/] attempt {attempt}/{RETRY_ATTEMPTS}")
            except requests.exceptions.RequestException as e:
                console.log(f"  [yellow]Request error attempt {attempt}:[/] {e}")

            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)

        console.log(f"  [red]✗ Failed after {RETRY_ATTEMPTS} attempts:[/] {url}")
        return None

    def _extract_html_text(self, html: str, url: str) -> str:
        """
        Parse HTML and extract main content text.
        Strips nav, header, footer, scripts — keeps the legislative content.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove boilerplate elements
        for tag in soup.find_all(["nav", "header", "footer", "script",
                                   "style", "aside", "noscript"]):
            tag.decompose()

        # Try to find the main content container
        main = (
            soup.find("main") or
            soup.find(id="content") or
            soup.find(class_=re.compile(r"content|main|body", re.I)) or
            soup.find("article") or
            soup.body
        )

        if main is None:
            return soup.get_text(separator="\n", strip=True)

        # Preserve section structure with newlines
        text = main.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace while preserving paragraph breaks
        import re as _re
        text = _re.sub(r'\n{4,}', '\n\n\n', text)
        text = _re.sub(r' {2,}', ' ', text)

        return text.strip()

    def _extract_pdf_text(self, pdf_bytes: bytes, url: str) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            import io
            from pypdf import PdfReader
            reader   = PdfReader(io.BytesIO(pdf_bytes))
            pages    = [page.extract_text() or "" for page in reader.pages]
            raw_text = "\n\n".join(pages)
            console.log(f"  [dim]PDF:[/] {len(reader.pages)} pages extracted from {url[-40:]}")
            return raw_text
        except Exception as e:
            console.log(f"  [red]PDF parse error:[/] {e}")
            return ""


# Import re at module level for _extract_html_text
import re
