"""
PlanIQ — Phase 1 v2: Full 25 Council Batch Ingestion
=====================================================
Updated registry covering all downloaded council development plans.

Special handling:
  - Meath: 2 volumes merged before chunking
  - Roscommon: 2 volumes merged before chunking
  - Kildare: 3 chapters (Housing, Rural Economy, Dev Management Standards)
             merged before chunking

Run:
  python phase1_ingest_v2.py                    # ingest all new councils
  python phase1_ingest_v2.py --council limerick # ingest one council
  python phase1_ingest_v2.py --test             # dry run
  python phase1_ingest_v2.py --skip-existing    # skip already-ingested councils
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

RAW_DATA_DIR     = Path(__file__).parent / "data" / "raw"
MIN_CHARS_PER_PAGE   = 50
OCR_TRIGGER_THRESHOLD = 100
BATCH_SIZE_PAGES     = 50

# ── Full 25-council registry ──────────────────────────────────────────────────
# Already ingested in phase1_ingest.py (batch 1):
#   dublin_city, fingal, south_dublin, dun_laoghaire,
#   cork_city, galway_city, cork_county
#
# New councils below — skip_if_exists=True prevents re-ingesting batch 1

COUNCIL_PLANS = {

    # ── Batch 1 (already ingested — skip) ────────────────────────────────────
    "dublin_city": {
        "title":          "Dublin City Development Plan 2022-2028",
        "files":          ["dublin_city_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.DUBLIN_CITY,
        "effective_date": date(2022, 10, 14),
        "act_year":       2022,
        "skip_if_exists": True,
    },
    "fingal": {
        "title":          "Fingal County Development Plan 2023-2029",
        "files":          ["fingal_devplan_2023.pdf"],
        "jurisdiction":   Jurisdiction.FINGAL,
        "effective_date": date(2023, 4, 5),
        "act_year":       2023,
        "skip_if_exists": True,
    },
    "south_dublin": {
        "title":          "South Dublin County Development Plan 2022-2028",
        "files":          ["south_dublin.pdf"],
        "jurisdiction":   Jurisdiction.SOUTH_DUBLIN,
        "effective_date": date(2022, 8, 3),
        "act_year":       2022,
        "skip_if_exists": True,
    },
    "dun_laoghaire": {
        "title":          "Dun Laoghaire-Rathdown Development Plan 2022-2028",
        "files":          ["dlr_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.DUN_LAOGHAIRE,
        "effective_date": date(2022, 4, 21),
        "act_year":       2022,
        "skip_if_exists": True,
    },
    "cork_city": {
        "title":          "Cork City Development Plan 2022-2028",
        "files":          ["cork_city_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.CORK_CITY,
        "effective_date": date(2022, 8, 8),
        "act_year":       2022,
        "skip_if_exists": True,
    },
    "galway_city": {
        "title":          "Galway City Development Plan 2023-2029",
        "files":          ["galway_city_devplan_2023.pdf"],
        "jurisdiction":   Jurisdiction.GALWAY_CITY,
        "effective_date": date(2023, 1, 4),
        "act_year":       2023,
        "skip_if_exists": True,
    },
    "cork_county": {
        "title":          "Cork County Development Plan 2022-2028",
        "files":          ["cork_county_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.CORK_COUNTY,
        "effective_date": date(2022, 6, 1),
        "act_year":       2022,
        "skip_if_exists": True,
    },

    # ── Batch 2 — new councils ─────────────────────────────────────────────
    "limerick": {
        "title":          "Limerick Development Plan 2022-2028",
        "files":          ["limerick_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.LIMERICK,
        "effective_date": date(2022, 5, 16),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "waterford": {
        "title":          "Waterford City and County Development Plan 2022-2028",
        "files":          ["waterford_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.WATERFORD,
        "effective_date": date(2022, 7, 15),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "kerry": {
        "title":          "Kerry County Development Plan 2022-2028",
        "files":          ["kerry_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.KERRY,
        "effective_date": date(2022, 6, 13),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "kildare": {
        "title":          "Kildare County Development Plan 2023-2029 (Chapters 3, 9, 15)",
        "files":          [
            "kildare_ch3.pdf",
            "kildare_ch9.pdf",
            "kildare_ch15.pdf",
        ],
        "jurisdiction":   Jurisdiction.KILDARE,
        "effective_date": date(2023, 2, 13),
        "act_year":       2023,
        "skip_if_exists": False,
        "notes":          "Housing, Rural Economy and Development Management Standards chapters only",
    },
    "meath": {
        "title":          "Meath County Development Plan 2021-2027 (Consolidated)",
        "files":          [
            "meath_devplan_2021_volume1.pdf",
            "meath_devplan_2021_volume2.pdf",
        ],
        "jurisdiction":   Jurisdiction.MEATH,
        "effective_date": date(2021, 10, 25),
        "act_year":       2021,
        "skip_if_exists": False,
        "notes":          "Volumes 1 and 2 merged — includes Variations 1, 2 and 3",
    },
    "wicklow": {
        "title":          "Wicklow County Development Plan 2022-2028",
        "files":          ["wicklow_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.WICKLOW,
        "effective_date": date(2022, 4, 4),
        "act_year":       2022,
        "skip_if_exists": False,
    },

    # ── Batch 3 ────────────────────────────────────────────────────────────
    "wexford": {
        "title":          "Wexford County Development Plan 2022-2028",
        "files":          ["wexford_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.WEXFORD,
        "effective_date": date(2022, 7, 25),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "kilkenny": {
        "title":          "Kilkenny City and County Development Plan 2021-2027",
        "files":          ["kilkenny_devplan_2021.pdf"],
        "jurisdiction":   Jurisdiction.KILKENNY,
        "effective_date": date(2021, 10, 15),
        "act_year":       2021,
        "skip_if_exists": False,
    },
    "tipperary": {
        "title":          "Tipperary County Development Plan 2022-2028",
        "files":          ["tipperary_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.TIPPERARY,
        "effective_date": date(2022, 8, 22),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "laois": {
        "title":          "Laois County Development Plan 2021-2027",
        "files":          ["laois_devplan_2021.pdf"],
        "jurisdiction":   Jurisdiction.LAOIS,
        "effective_date": date(2022, 3, 8),
        "act_year":       2021,
        "skip_if_exists": False,
    },
    "longford": {
        "title":          "Longford County Development Plan 2021-2027",
        "files":          ["longford_devplan_2021.pdf"],
        "jurisdiction":   Jurisdiction.LONGFORD,
        "effective_date": date(2021, 12, 13),
        "act_year":       2021,
        "skip_if_exists": False,
    },
    "louth": {
        "title":          "Louth County Development Plan 2021-2027 (Consolidated incl. Variation 1)",
        "files":          ["louth_devplan_2021.pdf"],
        "jurisdiction":   Jurisdiction.LOUTH,
        "effective_date": date(2021, 10, 4),
        "act_year":       2021,
        "skip_if_exists": False,
    },
    "monaghan": {
        "title":          "Monaghan County Development Plan 2025-2031",
        "files":          ["monaghan_devplan_2025.pdf"],
        "jurisdiction":   Jurisdiction.MONAGHAN,
        "effective_date": date(2025, 1, 1),
        "act_year":       2025,
        "skip_if_exists": False,
        "notes":          "Most recent plan — supersedes 2019-2025",
    },
    "roscommon": {
        "title":          "Roscommon County Development Plan 2022-2028",
        "files":          [
            "roscommon_devplan_2023_volume1.pdf",
            "roscommon_devplan_2023_volume2.pdf",
        ],
        "jurisdiction":   Jurisdiction.ROSCOMMON,
        "effective_date": date(2022, 4, 19),
        "act_year":       2022,
        "skip_if_exists": False,
        "notes":          "Volumes 1 and 2 merged",
    },
    "clare": {
        "title":          "Clare County Development Plan 2023-2029",
        "files":          ["clare_devplan_2023.pdf"],
        "jurisdiction":   Jurisdiction.CLARE,
        "effective_date": date(2023, 6, 1),
        "act_year":       2023,
        "skip_if_exists": False,
    },
    "galway_county": {
        "title":          "Galway County Development Plan 2022-2028",
        "files":          ["galway_county_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.GALWAY_COUNTY,
        "effective_date": date(2022, 6, 20),
        "act_year":       2022,
        "skip_if_exists": False,
    },
    "cavan": {
        "title":          "Cavan County Development Plan 2022-2028",
        "files":          ["cavan_devplan_2022.pdf"],
        "jurisdiction":   Jurisdiction.CAVAN,
        "effective_date": date(2022, 7, 11),
        "act_year":       2022,
        "skip_if_exists": False,
        "notes":          "Includes Local Area Plan for Cavan Town. Draft Variation No. 1 in progress 2026.",
    },
    "offaly": {
        "title":          "Offaly County Development Plan 2021-2027",
        "files":          ["offaly_devplan_2021.pdf"],
        "jurisdiction":   Jurisdiction.OFFALY,
        "effective_date": date(2021, 10, 1),
        "act_year":       2021,
        "skip_if_exists": False,
    },
}


# ── PDF extraction (same as phase1_ingest.py) ─────────────────────────────────

def _clean_page_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[\|\s\d]+$', line) and len(line) < 10:
            continue
        if re.match(r'^[-_=|]{3,}$', line):
            continue
        if len(line) < 4:
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def extract_text_from_pdf(pdf_path: Path, label: str) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        reader    = PdfReader(str(pdf_path))
        total     = len(reader.pages)
        sample    = min(10, total)
        sample_tx = "".join(reader.pages[i].extract_text() or "" for i in range(sample))
        avg_chars = len(sample_tx) / sample if sample else 0

        if avg_chars < OCR_TRIGGER_THRESHOLD:
            console.log(f"  [yellow]OCR triggered for {pdf_path.name}[/]")
            return _extract_with_ocr(pdf_path, total), "ocr"

        all_text  = []
        skipped   = 0
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
        console.log(f"  [red]Extraction error: {e}[/]")
        return "", "error"


def _extract_with_ocr(pdf_path: Path, total_pages: int) -> str:
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
            task = prog.add_task("", total=total_pages)
            for n in range(total_pages):
                try:
                    pix  = doc[n].get_pixmap(matrix=fitz.Matrix(200/72, 200/72), alpha=False)
                    img  = Image.open(io.BytesIO(pix.tobytes("png")))
                    t    = _clean_page_text(
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


def extract_multi_file(files: list[Path], label: str) -> tuple[str, str]:
    """Merge text from multiple PDFs (Meath, Roscommon, Kildare)."""
    all_parts = []
    method    = "pypdf"
    for f in files:
        if not f.exists():
            console.log(f"  [red]Missing: {f.name}[/]")
            continue
        console.log(f"  [dim]Reading {f.name} ({f.stat().st_size/1024/1024:.1f} MB)[/]")
        text, m = extract_text_from_pdf(f, f.stem)
        if m == "ocr":
            method = "ocr"
        if text:
            all_parts.append(text)
    return "\n\n".join(all_parts), method


# ── Ingest one council ────────────────────────────────────────────────────────

def ingest_council(key: str, plan: dict, kb, dry_run: bool) -> dict:
    stats = {
        "council": key, "title": plan["title"],
        "status": "pending", "extraction": "",
        "raw_chars": 0, "chunks_created": 0, "chunks_added": 0,
        "errors": [],
    }

    files = [RAW_DATA_DIR / f for f in plan["files"]]

    # Check all files exist
    missing = [f for f in files if not f.exists()]
    if missing:
        stats["status"] = "missing"
        stats["errors"] = [f"Missing: {f.name}" for f in missing]
        console.log(f"  [red]✗ Missing files: {[f.name for f in missing]}[/]")
        return stats

    console.log(f"\n[cyan]Processing:[/] {plan['title'][:65]}")
    if plan.get("notes"):
        console.log(f"  [dim]{plan['notes']}[/]")

    # Extract
    start = time.time()
    if len(files) > 1:
        raw_text, method = extract_multi_file(files, key)
    else:
        console.log(f"  File: {files[0].name} ({files[0].stat().st_size/1024/1024:.1f} MB)")
        raw_text, method = extract_text_from_pdf(files[0], key)

    elapsed = time.time() - start

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
        document_type  = DocumentType.COUNCIL_DEVPLAN,
        jurisdiction   = plan["jurisdiction"],
        source_title   = plan["title"],
        source_url     = f"local://{plan['files'][0]}",
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

    added = kb.add_chunks(chunks)
    stats["chunks_added"] = added
    stats["status"]       = "success"
    console.log(f"  [green]✓[/] {len(chunks)} chunks | {added} added to KB")
    return stats


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(councils: list[str] = None, dry_run: bool = False,
        skip_existing: bool = True) -> list[dict]:

    console.print(Panel.fit(
        "[bold]PlanIQ — Phase 1 v2: Full 25-Council Ingestion[/]\n"
        "Expanding knowledge base to all downloaded Irish councils",
        border_style="cyan"
    ))

    if not dry_run:
        kb = PlanIQKnowledgeBase()
        console.log(f"[green]✓[/] KB ready — {kb.get_stats()['total_chunks_chroma']:,} existing chunks")
    else:
        kb = None
        console.log("[yellow]DRY RUN — no DB writes[/]")

    # Filter councils
    if councils:
        selected = {k: v for k, v in COUNCIL_PLANS.items() if k in councils}
    else:
        selected = COUNCIL_PLANS.copy()

    # Skip already-ingested batch 1 unless explicitly requested
    if skip_existing:
        selected = {k: v for k, v in selected.items()
                    if not v.get("skip_if_exists", False)}
        console.log("[dim]Skipping already-ingested Batch 1 councils[/]")

    console.log(f"Processing {len(selected)} councils: {list(selected.keys())}")

    all_stats = []
    for key, plan in selected.items():
        result = ingest_council(key, plan, kb, dry_run=dry_run)
        all_stats.append(result)

    # Summary
    console.rule("[cyan]Ingestion Complete[/]")
    table = Table(title="Results", border_style="dim")
    table.add_column("Council",        style="cyan", max_width=18)
    table.add_column("Status",         width=12)
    table.add_column("Method",         width=7)
    table.add_column("Raw chars",      justify="right")
    table.add_column("Chunks",         justify="right")
    table.add_column("Added",          justify="right")

    totals = {"chars": 0, "created": 0, "added": 0}
    STATUS = {
        "success":          "[green]success[/]",
        "missing":          "[red]missing[/]",
        "extraction_failed": "[red]failed[/]",
        "no_chunks":        "[yellow]no chunks[/]",
        "dry_run":          "[dim]dry run[/]",
    }
    for s in all_stats:
        table.add_row(
            s["council"],
            STATUS.get(s["status"], s["status"]),
            s.get("extraction", "-"),
            f"{s['raw_chars']:,}" if s['raw_chars'] else "-",
            str(s['chunks_created']) if s['chunks_created'] else "-",
            str(s['chunks_added'])   if s['chunks_added']   else "-",
        )
        totals["chars"]   += s["raw_chars"]
        totals["created"] += s["chunks_created"]
        totals["added"]   += s["chunks_added"]

    table.add_section()
    table.add_row("TOTAL", "", "",
                  f"{totals['chars']:,}",
                  str(totals["created"]),
                  str(totals["added"]), style="bold")
    console.print(table)

    if not dry_run and kb:
        st = kb.get_stats()
        console.print(
            f"\n[bold]Knowledge Base Total:[/] "
            f"[green]{st['total_chunks_chroma']:,}[/] chunks across "
            f"[green]{st['total_docs_ingested']}[/] documents"
        )

    errors = [(s["council"], e) for s in all_stats for e in s.get("errors", [])]
    if errors:
        console.print("\n[red]Errors:[/]")
        for c, e in errors:
            console.print(f"  {c}: {e}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlanIQ Phase 1 v2 — Full Council Ingestion")
    parser.add_argument("--council", nargs="+",
                        help=f"Specific councils. Options: {list(COUNCIL_PLANS.keys())}")
    parser.add_argument("--test",          action="store_true", help="Dry run")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip batch 1 councils already ingested (default: True)")
    parser.add_argument("--all",           action="store_true",
                        help="Include batch 1 councils (re-ingest everything)")
    args = parser.parse_args()

    skip = not args.all
    run(councils=args.council, dry_run=args.test, skip_existing=skip)
