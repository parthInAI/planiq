"""
PlanIQ — Ministerial Guidelines Ingestion
==========================================
Ingests 8 Section 28 ministerial guidelines into the knowledge base.

These guidelines apply nationally — every planning authority must have
regard to them when making decisions. They carry almost as much legal
weight as the Acts themselves.

Run:
  python ingest_guidelines.py              # ingest all guidelines
  python ingest_guidelines.py --test       # dry run
  python ingest_guidelines.py --guideline rural_housing  # single guideline
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

from ingestion.schema import DocumentType, Jurisdiction, ConfidenceLevel
from ingestion.chunker import SemanticChunker
from knowledge_base.store import PlanIQKnowledgeBase

console = Console()

RAW_DATA_DIR         = Path(__file__).parent / "data" / "raw"
MIN_CHARS_PER_PAGE   = 80
OCR_TRIGGER_THRESHOLD = 100
BATCH_SIZE_PAGES     = 50

# ── Guidelines registry ───────────────────────────────────────────────────────
GUIDELINES = {

    "rural_housing": {
        "title":          "Sustainable Rural Housing Guidelines for Planning Authorities 2005",
        "filename":       "rural_housing_guidelines_2005.pdf",
        "effective_date": date(2005, 4, 1),
        "act_year":       2005,
        "status":         "active",
        "notes":          "Most cited document in rural planning refusals. Defines genuine local need under NPO 19. Circular PL 2/2017 updated local needs criteria.",
        "key_topics":     ["rural housing", "genuine local need", "NPO 19", "one-off housing", "rural area types"],
    },

    "building_heights": {
        "title":          "Urban Development and Building Heights Guidelines for Planning Authorities 2018",
        "filename":       "building_heights_guidelines_2018.pdf",
        "effective_date": date(2018, 12, 1),
        "act_year":       2018,
        "status":         "active",
        "notes":          "Contains Specific Planning Policy Requirements (SPPRs) which override conflicting development plan policies. Default height 6 storeys town centre, 4 storeys suburban.",
        "key_topics":     ["building height", "SPPR", "urban development", "storeys", "town centre"],
    },

    "sustainable_residential": {
        "title":          "Sustainable Residential Development and Compact Settlements Guidelines 2024",
        "filename":       "sustainable_residential_guidelines_2024.pdf",
        "effective_date": date(2024, 1, 15),
        "act_year":       2024,
        "status":         "active",
        "notes":          "Replaces and revokes 2009 Sustainable Residential Developments in Urban Areas guidelines. Sets density standards, housing design standards, placemaking guidance.",
        "key_topics":     ["residential density", "compact settlements", "housing standards", "placemaking", "urban design"],
    },

    "apartment_design": {
        "title":          "Design Standards for Apartments Guidelines for Planning Authorities 2025",
        "filename":       "apartment_design_guidelines_2022.pdf",
        "effective_date": date(2025, 7, 1),
        "act_year":       2025,
        "status":         "active",
        "notes":          "Most recent version — replaces all previous apartment guidelines including 2022 and 2023 versions. Sets minimum floor areas, dual aspect requirements, amenity space standards.",
        "key_topics":     ["apartments", "floor area", "dual aspect", "amenity space", "build-to-rent", "storage"],
    },

    "development_plans": {
        "title":          "Development Plans Guidelines for Planning Authorities 2022",
        "filename":       "development_plans_guidelines_2022.pdf",
        "effective_date": date(2022, 7, 1),
        "act_year":       2022,
        "status":         "active",
        "notes":          "Guidance on preparation and content of development plans. Core strategy requirements, zoning, variation procedures.",
        "key_topics":     ["development plan", "core strategy", "zoning", "variation", "local area plan"],
    },

    "retail_planning": {
        "title":          "Retail Planning Guidelines for Planning Authorities 2012",
        "filename":       "retail_planning_guidelines_2012.pdf",
        "effective_date": date(2012, 4, 1),
        "act_year":       2012,
        "status":         "active",
        "notes":          "Sequential approach to retail development, impact assessments, floor area thresholds, town centre first policy.",
        "key_topics":     ["retail", "sequential test", "town centre", "floor area threshold", "retail impact assessment"],
    },

    "flood_risk": {
        "title":          "The Planning System and Flood Risk Management Guidelines 2009",
        "filename":       "flood_risk_guidelines_2009.pdf",
        "effective_date": date(2009, 11, 1),
        "act_year":       2009,
        "status":         "active",
        "notes":          "Flood Zone A (high risk), Zone B (moderate risk), Zone C (low risk). Justification Test for development in flood zones. SuDS requirements.",
        "key_topics":     ["flood risk", "flood zone", "justification test", "SuDS", "flood plain", "OPW"],
    },

    "childcare": {
        "title":          "Childcare Facilities Guidelines for Planning Authorities 2001",
        "filename":       "childcare_guidelines_2001.pdf",
        "effective_date": date(2001, 6, 1),
        "act_year":       2001,
        "status":         "active",
        "notes":          "One childcare facility per 75 dwellings in new housing areas. Site assessment criteria, outdoor play area requirements.",
        "key_topics":     ["childcare", "creche", "pre-school", "75 dwellings threshold", "outdoor play area"],
    },
}


# ── PDF extraction ────────────────────────────────────────────────────────────

def _clean_page_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if re.match(r'^[\|\s\d]+$', stripped) and len(stripped) < 10:
            continue
        if re.match(r'^[-_=|]{3,}$', stripped):
            continue
        if len(stripped) < 4:
            continue
        cleaned.append(stripped)
    result = "\n".join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'\n[ \t]+\n', '\n\n', result)
    result = re.sub(r'([.!?])\n([A-Z])', r'\1\n\n\2', result)
    return result.strip()


def extract_text(pdf_path: Path, label: str) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        reader    = PdfReader(str(pdf_path))
        total     = len(reader.pages)
        sample    = "".join(reader.pages[i].extract_text() or "" for i in range(min(10, total)))
        avg_chars = len(sample) / min(10, total) if total else 0

        console.log(f"  [dim]{total} pages | avg {avg_chars:.0f} chars/page[/]")

        if avg_chars < OCR_TRIGGER_THRESHOLD:
            console.log("  [yellow]OCR triggered[/]")
            return _extract_with_ocr(pdf_path, total), "ocr"

        all_text = []
        skipped  = 0
        with Progress(SpinnerColumn(), TextColumn(f"  {label}..."),
                      BarColumn(), TextColumn("{task.percentage:.0f}%"),
                      console=console) as prog:
            task = prog.add_task("", total=total)
            for i in range(0, total, BATCH_SIZE_PAGES):
                for j in range(i, min(i + BATCH_SIZE_PAGES, total)):
                    t = _clean_page_text(reader.pages[j].extract_text() or "")
                    if len(t) >= MIN_CHARS_PER_PAGE:
                        all_text.append(t)
                    else:
                        skipped += 1
                    prog.advance(task)

        console.log(f"  [green]pypdf:[/] {total - skipped} pages | {skipped} skipped")
        return "\n\n".join(all_text), "pypdf"

    except Exception as e:
        console.log(f"  [red]Error: {e}[/]")
        return "", "error"


def _extract_with_ocr(pdf_path: Path, total: int) -> str:
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
        doc      = fitz.open(str(pdf_path))
        all_text = []
        skipped  = 0
        with Progress(SpinnerColumn(), TextColumn("  OCR..."),
                      BarColumn(), console=console) as prog:
            task = prog.add_task("", total=total)
            for n in range(total):
                try:
                    pix = doc[n].get_pixmap(matrix=fitz.Matrix(200/72, 200/72), alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    t   = _clean_page_text(
                        pytesseract.image_to_string(img, lang="eng", config="--psm 1 --oem 3")
                    )
                    if len(t) >= MIN_CHARS_PER_PAGE:
                        all_text.append(t)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                prog.advance(task)
        doc.close()
        return "\n\n".join(all_text)
    except ImportError as e:
        console.log(f"  [red]OCR deps missing: {e}[/]")
        return ""


# ── Ingest one guideline ──────────────────────────────────────────────────────

def ingest_guideline(key: str, g: dict, kb, dry_run: bool) -> dict:
    stats = {
        "key":            key,
        "title":          g["title"][:55],
        "status":         "pending",
        "extraction":     "",
        "raw_chars":      0,
        "chunks_created": 0,
        "chunks_added":   0,
        "errors":         [],
    }

    pdf_path = RAW_DATA_DIR / g["filename"]

    if not pdf_path.exists():
        stats["status"] = "missing"
        stats["errors"].append(f"File not found: {g['filename']}")
        console.log(f"  [red]✗ Missing:[/] {g['filename']}")
        return stats

    size_mb = pdf_path.stat().st_size / 1024 / 1024
    console.log(f"\n[cyan]Processing:[/] {g['title'][:65]}")
    console.log(f"  File: {g['filename']} ({size_mb:.1f} MB) | Status: {g['status']}")
    if g.get("notes"):
        console.log(f"  [dim]{g['notes'][:100]}[/]")

    start    = time.time()
    raw_text, method = extract_text(pdf_path, key)
    elapsed  = time.time() - start

    if not raw_text:
        stats["status"] = "extraction_failed"
        stats["errors"].append("No text extracted")
        return stats

    stats["extraction"] = method
    stats["raw_chars"]  = len(raw_text)
    console.log(f"  [green]✓[/] {len(raw_text):,} chars via {method} in {elapsed:.1f}s")

    if dry_run:
        console.log("  [dim]DRY RUN — skipping chunking and DB write[/]")
        stats["status"] = "dry_run"
        return stats

    # Chunk
    chunker = SemanticChunker(
        document_type  = DocumentType.MINISTERIAL_GUIDE,
        jurisdiction   = Jurisdiction.NATIONAL,
        source_title   = g["title"],
        source_url     = f"local://{g['filename']}",
        act_year       = g["act_year"],
        effective_date = g["effective_date"],
        confidence     = ConfidenceLevel.HIGH,
        is_verbatim    = True,
    )
    chunks = chunker.chunk(raw_text)
    stats["chunks_created"] = len(chunks)

    if not chunks:
        stats["status"] = "no_chunks"
        stats["errors"].append("Chunker produced 0 chunks")
        return stats

    added = kb.add_chunks(chunks)
    stats["chunks_added"] = added
    stats["status"]       = "success"
    console.log(f"  [green]✓[/] {len(chunks)} chunks | {added} added to KB")
    return stats


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(guidelines: list[str] = None, dry_run: bool = False) -> list[dict]:

    console.print(Panel.fit(
        "[bold]PlanIQ — Ministerial Guidelines Ingestion[/]\n"
        "8 Section 28 guidelines — nationally binding on all planning authorities",
        border_style="cyan"
    ))

    if not dry_run:
        kb = PlanIQKnowledgeBase()
        console.log(f"[green]✓[/] KB ready — {kb.get_stats()['total_chunks_chroma']:,} existing chunks")
    else:
        kb = None
        console.log("[yellow]DRY RUN — no DB writes[/]")

    selected = {k: v for k, v in GUIDELINES.items()
                if not guidelines or k in guidelines}

    console.log(f"Processing {len(selected)} guidelines: {list(selected.keys())}")

    all_stats = []
    for key, g in selected.items():
        result = ingest_guideline(key, g, kb, dry_run=dry_run)
        all_stats.append(result)

    # Summary table
    console.rule("[cyan]Ingestion Complete[/]")
    table = Table(title="Ministerial Guidelines Results", border_style="dim")
    table.add_column("Guideline",      style="cyan", max_width=20)
    table.add_column("Status",         width=12)
    table.add_column("Method",         width=7)
    table.add_column("Raw chars",      justify="right")
    table.add_column("Chunks",         justify="right")
    table.add_column("Added",          justify="right")

    STATUS = {
        "success":           "[green]success[/]",
        "missing":           "[red]missing[/]",
        "extraction_failed": "[red]failed[/]",
        "no_chunks":         "[yellow]no chunks[/]",
        "dry_run":           "[dim]dry run[/]",
    }

    totals = {"chars": 0, "created": 0, "added": 0}
    for s in all_stats:
        table.add_row(
            s["key"],
            STATUS.get(s["status"], s["status"]),
            s.get("extraction", "-"),
            f"{s['raw_chars']:,}" if s["raw_chars"] else "-",
            str(s["chunks_created"]) if s["chunks_created"] else "-",
            str(s["chunks_added"])   if s["chunks_added"]   else "-",
        )
        totals["chars"]   += s["raw_chars"]
        totals["created"] += s["chunks_created"]
        totals["added"]   += s["chunks_added"]

    table.add_section()
    table.add_row(
        "TOTAL", "", "",
        f"{totals['chars']:,}",
        str(totals["created"]),
        str(totals["added"]),
        style="bold",
    )
    console.print(table)

    if not dry_run and kb:
        st = kb.get_stats()
        console.print(
            f"\n[bold]Knowledge Base Total:[/] "
            f"[green]{st['total_chunks_chroma']:,}[/] chunks across "
            f"[green]{st['total_docs_ingested']}[/] documents"
        )

    errors = [(s["key"], e) for s in all_stats for e in s.get("errors", [])]
    if errors:
        console.print("\n[red]Errors:[/]")
        for k, e in errors:
            console.print(f"  {k}: {e}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlanIQ — Ministerial Guidelines Ingestion")
    parser.add_argument("--guideline", nargs="+",
                        help=f"Specific guidelines. Options: {list(GUIDELINES.keys())}")
    parser.add_argument("--test", action="store_true", help="Dry run — no DB writes")
    args = parser.parse_args()

    run(guidelines=args.guideline, dry_run=args.test)
