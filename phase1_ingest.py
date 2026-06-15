"""
PlanIQ — Phase 1: Batch Council Development Plan Ingestion
===========================================================
Ingests all 7 council development plan PDFs into the knowledge base.

New capabilities added in Phase 1:
  1. OCR fallback — if pypdf extracts < 100 chars per page on average,
     automatically switch to pytesseract OCR (handles scanned PDFs like DLR)
  2. Blank page filter — skips pages that are maps, images, or blank
     (less than 50 meaningful characters after cleaning)
  3. Large file handling — processes big PDFs (Fingal 86MB) page by page
     in batches to avoid memory crashes
  4. Per-council progress tracking — shows exactly how many chunks
     each plan contributed
  5. Multi-source registration — all 7 councils registered in one registry

Run:
  python phase1_ingest.py                    # ingest all councils
  python phase1_ingest.py --council fingal   # ingest one council only
  python phase1_ingest.py --test            # dry run, no DB writes
"""

import sys
import re
import argparse
import time
from pathlib import Path
from datetime import date
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent / "knowledge_base"))

from ingestion.schema import DocumentType, Jurisdiction, ConfidenceLevel, PlanningChunk
from ingestion.chunker import SemanticChunker
from knowledge_base.store import PlanIQKnowledgeBase

console = Console()

RAW_DATA_DIR = Path(__file__).parent / "data" / "raw"

# ── Blank page detection ──────────────────────────────────────────────────────
MIN_CHARS_PER_PAGE = 50       # Pages with fewer chars than this are skipped
OCR_TRIGGER_THRESHOLD = 100   # If avg chars/page < this, switch to OCR mode
BATCH_SIZE_PAGES = 50         # Process large PDFs in batches of 50 pages


# ── Council registry — all 7 Batch 1 plans ───────────────────────────────────
COUNCIL_PLANS = {

    "dublin_city": {
        "title":         "Dublin City Development Plan 2022-2028 — Written Statement",
        "filename":      "dublin_city_devplan_2022.pdf",
        "jurisdiction":  Jurisdiction.DUBLIN_CITY,
        "effective_date": date(2022, 10, 14),
        "act_year":      2022,
        "priority":      1,
        "notes":         "Already ingested in Step 1 — will skip duplicates automatically",
    },

    "fingal": {
        "title":         "Fingal County Development Plan 2023-2029 — Written Statement",
        "filename":      "fingal_devplan_2023.pdf",
        "jurisdiction":  Jurisdiction.FINGAL,
        "effective_date": date(2023, 4, 5),
        "act_year":      2023,
        "priority":      1,
        "notes":         "86MB — processed in page batches",
    },

    "south_dublin": {
        "title":         "South Dublin County Development Plan 2022-2028 — Written Statement",
        "filename":      "south_dublin.pdf",
        "jurisdiction":  Jurisdiction.SOUTH_DUBLIN,
        "effective_date": date(2022, 8, 3),
        "act_year":      2022,
        "priority":      1,
        "notes":         "Variation No.1 (Clondalkin) made March 2026 — check for updates",
    },

    "dun_laoghaire": {
        "title":         "Dun Laoghaire-Rathdown County Development Plan 2022-2028 — Written Statement",
        "filename":      "dlr_devplan_2022.pdf",
        "jurisdiction":  Jurisdiction.DUN_LAOGHAIRE,
        "effective_date": date(2022, 4, 21),
        "act_year":      2022,
        "priority":      1,
        "notes":         "6MB for 380 pages — likely scanned, OCR will activate",
    },

    "cork_city": {
        "title":         "Cork City Development Plan 2022-2028 — Written Statement Volume 1",
        "filename":      "cork_city_devplan_2022.pdf",
        "jurisdiction":  Jurisdiction.CORK_CITY,
        "effective_date": date(2022, 8, 8),
        "act_year":      2022,
        "priority":      2,
        "notes":         "51MB — subject to Ministerial Direction on certain sections",
    },

    "galway_city": {
        "title":         "Galway City Development Plan 2023-2029 — Written Statement (as amended May 2025)",
        "filename":      "galway_city_devplan_2023.pdf",
        "jurisdiction":  Jurisdiction.GALWAY_CITY,
        "effective_date": date(2023, 1, 4),
        "act_year":      2023,
        "priority":      2,
        "notes":         "Amended by High Court Order May 2025 and Variation No.1 Feb 2026",
    },

    "cork_county": {
        "title":         "Cork County Development Plan 2022-2028 — Written Statement",
        "filename":      "cork_county_devplan_2022.pdf",
        "jurisdiction":  Jurisdiction.CORK_COUNTY,
        "effective_date": date(2022, 6, 1),
        "act_year":      2022,
        "priority":      2,
        "notes":         "Variation No.1 proposed March 2026 — monitor for adoption",
    },
}


# ── PDF text extraction with OCR fallback ────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path, council_key: str) -> tuple[str, str]:
    """
    Extract text from a PDF.
    Returns (text, method) where method is 'pypdf' or 'ocr'.

    Strategy:
      1. Try pypdf first (fast, accurate for machine-readable PDFs)
      2. Sample first 10 pages — if avg chars < OCR_TRIGGER_THRESHOLD, switch to OCR
      3. OCR uses pymupdf to render pages as images + pytesseract to read them
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        console.log(f"  [dim]PDF opened: {total_pages} pages[/]")

        # ── Sample first 10 pages to detect if OCR is needed ──────────────
        sample_pages = min(10, total_pages)
        sample_text = ""
        for i in range(sample_pages):
            page_text = reader.pages[i].extract_text() or ""
            sample_text += page_text

        avg_chars = len(sample_text) / sample_pages if sample_pages > 0 else 0
        console.log(f"  [dim]Avg chars/page (sample): {avg_chars:.0f}[/]")

        if avg_chars < OCR_TRIGGER_THRESHOLD:
            console.log(f"  [yellow]Low text density detected — switching to OCR mode[/]")
            return _extract_with_ocr(pdf_path, total_pages), "ocr"

        # ── Full pypdf extraction in batches ──────────────────────────────
        all_text = []
        skipped_pages = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"  Extracting {council_key}..."),
            BarColumn(),
            TextColumn("{task.percentage:.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=total_pages)

            for batch_start in range(0, total_pages, BATCH_SIZE_PAGES):
                batch_end = min(batch_start + BATCH_SIZE_PAGES, total_pages)
                batch_text = []

                for i in range(batch_start, batch_end):
                    page_text = reader.pages[i].extract_text() or ""
                    cleaned   = _clean_page_text(page_text)

                    if len(cleaned) < MIN_CHARS_PER_PAGE:
                        skipped_pages += 1
                        progress.advance(task)
                        continue

                    batch_text.append(cleaned)
                    progress.advance(task)

                all_text.extend(batch_text)

        console.log(
            f"  [green]pypdf:[/] {total_pages - skipped_pages} pages extracted "
            f"| {skipped_pages} blank/image pages skipped"
        )
        return "\n\n".join(all_text), "pypdf"

    except Exception as e:
        console.log(f"  [red]PDF extraction error: {e}[/]")
        return "", "error"


def _extract_with_ocr(pdf_path: Path, total_pages: int) -> str:
    """
    OCR extraction using pymupdf + pytesseract.
    Renders each page as an image at 200 DPI then reads it with tesseract.
    Used automatically for scanned PDFs like DLR.
    """
    try:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        all_text = []
        skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("  OCR processing..."),
            BarColumn(),
            TextColumn("{task.percentage:.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=total_pages)

            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    # Render at 200 DPI — good balance of quality vs speed
                    mat  = fitz.Matrix(200/72, 200/72)
                    pix  = page.get_pixmap(matrix=mat, alpha=False)
                    img  = Image.open(io.BytesIO(pix.tobytes("png")))

                    # tesseract with Irish English config
                    page_text = pytesseract.image_to_string(
                        img,
                        lang="eng",
                        config="--psm 1 --oem 3",
                    )
                    cleaned = _clean_page_text(page_text)

                    if len(cleaned) >= MIN_CHARS_PER_PAGE:
                        all_text.append(cleaned)
                    else:
                        skipped += 1

                except Exception as page_err:
                    console.log(f"  [yellow]OCR skip page {page_num}: {page_err}[/]")
                    skipped += 1

                progress.advance(task)

        doc.close()
        console.log(
            f"  [green]OCR:[/] {total_pages - skipped} pages extracted "
            f"| {skipped} skipped"
        )
        return "\n\n".join(all_text)

    except ImportError as e:
        console.log(f"  [red]OCR dependencies missing: {e}[/]")
        console.log("  Run: pip install pymupdf pytesseract Pillow")
        console.log("  Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        return ""


def _clean_page_text(text: str) -> str:
    """
    Clean extracted page text:
    - Remove excessive whitespace
    - Remove page numbers and headers
    - Remove lines that are just numbers or single characters
    - Preserve paragraph structure
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip blank lines
        if not line:
            continue

        # Skip lines that are just page numbers (e.g. "| 47 |" or "47")
        if re.match(r'^[\|\s\d]+$', line) and len(line) < 10:
            continue

        # Skip lines that are just dashes or underscores (decorative)
        if re.match(r'^[-_=|]{3,}$', line):
            continue

        # Skip very short lines that are likely headers/footers
        if len(line) < 4:
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    # Collapse multiple blank lines into double newline (paragraph separator)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_council(
    council_key: str,
    plan: dict,
    kb: PlanIQKnowledgeBase,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single council development plan.
    Returns stats dict for reporting.
    """
    stats = {
        "council":       council_key,
        "title":         plan["title"],
        "status":        "pending",
        "extraction":    "",
        "raw_chars":     0,
        "chunks_created": 0,
        "chunks_added":  0,
        "errors":        [],
    }

    pdf_path = RAW_DATA_DIR / plan["filename"]

    # Check file exists
    if not pdf_path.exists():
        stats["status"] = "missing"
        stats["errors"].append(f"File not found: {pdf_path.name}")
        console.log(f"  [red]✗ Missing:[/] {plan['filename']}")
        return stats

    console.log(f"\n[cyan]Processing:[/] {plan['title'][:60]}")
    console.log(f"  File: {plan['filename']} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if plan.get("notes"):
        console.log(f"  [dim]Note: {plan['notes']}[/]")

    # ── Extract text ──────────────────────────────────────────────────────
    start = time.time()
    raw_text, method = extract_text_from_pdf(pdf_path, council_key)
    elapsed = time.time() - start

    if not raw_text:
        stats["status"] = "extraction_failed"
        stats["errors"].append("No text extracted from PDF")
        return stats

    stats["extraction"] = method
    stats["raw_chars"]  = len(raw_text)
    console.log(
        f"  [green]✓[/] Extracted {len(raw_text):,} chars "
        f"via {method} in {elapsed:.1f}s"
    )

    if dry_run:
        console.log("  [dim]DRY RUN — skipping chunking and DB write[/]")
        stats["status"] = "dry_run"
        return stats

    # ── Chunk ─────────────────────────────────────────────────────────────
    chunker = SemanticChunker(
        document_type  = DocumentType.COUNCIL_DEVPLAN,
        jurisdiction   = plan["jurisdiction"],
        source_title   = plan["title"],
        source_url     = f"local://{plan['filename']}",
        act_year       = plan["act_year"],
        effective_date = plan["effective_date"],
        confidence     = ConfidenceLevel.HIGH,
        is_verbatim    = True,
    )

    chunks = chunker.chunk(raw_text)
    stats["chunks_created"] = len(chunks)

    if not chunks:
        stats["status"] = "no_chunks"
        stats["errors"].append("Chunker produced 0 chunks")
        return stats

    # ── Write to KB ───────────────────────────────────────────────────────
    added = kb.add_chunks(chunks)
    stats["chunks_added"] = added
    stats["status"]       = "success"

    console.log(
        f"  [green]✓[/] {len(chunks)} chunks created | "
        f"{added} new chunks added to KB"
    )

    return stats


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_phase1(
    councils: list[str] = None,
    dry_run:  bool = False,
) -> list[dict]:
    """
    Run Phase 1 ingestion for all or selected councils.
    """
    console.print(Panel.fit(
        "[bold]PlanIQ — Phase 1: Council Development Plans Ingestion[/]\n"
        "Expanding knowledge base to 7 Irish local authorities",
        border_style="cyan"
    ))

    # ── Initialise KB ─────────────────────────────────────────────────────
    if not dry_run:
        kb = PlanIQKnowledgeBase()
        console.log(f"[green]✓[/] KB ready — {kb.get_stats()['total_chunks_chroma']} existing chunks")
    else:
        kb = None
        console.log("[yellow]DRY RUN MODE — no DB writes[/]")

    # ── Select councils ───────────────────────────────────────────────────
    if councils:
        selected = {k: v for k, v in COUNCIL_PLANS.items() if k in councils}
        unknown  = [c for c in councils if c not in COUNCIL_PLANS]
        if unknown:
            console.log(f"[yellow]Unknown councils: {unknown}[/]")
            console.log(f"Valid options: {list(COUNCIL_PLANS.keys())}")
    else:
        # Sort by priority — do Priority 1 councils first
        selected = dict(sorted(
            COUNCIL_PLANS.items(),
            key=lambda x: x[1]["priority"]
        ))

    console.log(f"Processing {len(selected)} councils: {list(selected.keys())}")

    # ── Ingest each council ───────────────────────────────────────────────
    all_stats = []
    for council_key, plan in selected.items():
        result = ingest_council(council_key, plan, kb, dry_run=dry_run)
        all_stats.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    console.rule("[cyan]Phase 1 Complete[/]")

    table = Table(title="Ingestion Results", border_style="dim")
    table.add_column("Council",         style="cyan",  max_width=20)
    table.add_column("Status",          style="white", width=12)
    table.add_column("Method",          width=8)
    table.add_column("Raw chars",       justify="right")
    table.add_column("Chunks created",  justify="right")
    table.add_column("Chunks added",    justify="right")

    total_chars   = 0
    total_created = 0
    total_added   = 0

    for s in all_stats:
        status_style = {
            "success":          "[green]success[/]",
            "missing":          "[red]missing[/]",
            "extraction_failed": "[red]failed[/]",
            "no_chunks":        "[yellow]no chunks[/]",
            "dry_run":          "[dim]dry run[/]",
        }.get(s["status"], s["status"])

        table.add_row(
            s["council"],
            status_style,
            s.get("extraction", "-"),
            f"{s['raw_chars']:,}" if s['raw_chars'] else "-",
            str(s['chunks_created']) if s['chunks_created'] else "-",
            str(s['chunks_added'])   if s['chunks_added']   else "-",
        )
        total_chars   += s["raw_chars"]
        total_created += s["chunks_created"]
        total_added   += s["chunks_added"]

    table.add_section()
    table.add_row(
        "TOTAL", "", "",
        f"{total_chars:,}",
        str(total_created),
        str(total_added),
        style="bold",
    )
    console.print(table)

    # KB final stats
    if not dry_run and kb:
        kb_stats = kb.get_stats()
        console.print(
            f"\n[bold]Knowledge Base Total:[/] "
            f"[green]{kb_stats['total_chunks_chroma']:,}[/] chunks across "
            f"[green]{kb_stats['total_docs_ingested']}[/] documents"
        )

    # Print errors if any
    errors = [(s["council"], e) for s in all_stats for e in s.get("errors", [])]
    if errors:
        console.print("\n[red]Errors:[/]")
        for council, err in errors:
            console.print(f"  {council}: {err}")

    return all_stats


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlanIQ Phase 1 — Council Plan Ingestion")
    parser.add_argument(
        "--council", nargs="+",
        help=f"Specific councils to ingest. Options: {list(COUNCIL_PLANS.keys())}"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Dry run — extract and chunk but do not write to DB"
    )
    args = parser.parse_args()

    run_phase1(councils=args.council, dry_run=args.test)
