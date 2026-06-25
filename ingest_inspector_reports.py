"""
PlanIQ — An Coimisiún Pleanála Inspector Reports Ingestion
===========================================================
Ingests inspector reports into the knowledge base.

These reports show how Irish planning law is applied in practice —
the most valuable knowledge source after the legislation itself.

Run:
  python ingest_inspector_reports.py          # ingest all reports
  python ingest_inspector_reports.py --test   # dry run
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

REPORTS_DIR          = Path(__file__).parent / "data" / "raw" / "inspector_reports"
MIN_CHARS_PER_PAGE   = 80
OCR_TRIGGER_THRESHOLD = 100
BATCH_SIZE_PAGES     = 50

# ── Report registry ───────────────────────────────────────────────────────────
# Category codes:
#   RH  = Rural housing / genuine local need
#   RE  = Residential extension / exempted development
#   RD  = Residential development
#   APT = Apartment development
#   COM = Commercial / retail

REPORTS = {
    "r303159": {
        "title":       "ABP-303159-18 — Rural Housing, Donegal, Drainage Refused, Scenic Amenity",
        "filename":    "r303159.pdf",
        "jurisdiction": Jurisdiction.DONEGAL,
        "category":    "RH",
        "year":        2018,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "genuine local need", "drainage", "scenic amenity", "flood risk"],
        "notes":       "Refused on drainage and flood risk grounds. Donegal rural housing policy.",
    },
    "r304445": {
        "title":       "ABP-304445-19 — Rural Housing, Cork, Genuine Local Need Refused, Access Road",
        "filename":    "r304445.pdf",
        "jurisdiction": Jurisdiction.CORK_COUNTY,
        "category":    "RH",
        "year":        2019,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "genuine local need", "NPO 19", "access road", "sight lines"],
        "notes":       "Refused — applicant failed genuine local need test under NPO 19 and Rural Housing Guidelines 2005.",
    },
    "r307802": {
        "title":       "ABP-307802-20 — Residential Extension, Class 1 Exemption, Window Condition",
        "filename":    "r307802.pdf",
        "jurisdiction": Jurisdiction.NATIONAL,
        "category":    "RE",
        "year":        2020,
        "outcome":     "granted",
        "key_issues":  ["Class 1", "exempted development", "rear extension", "window condition 6a", "unauthorised development"],
        "notes":       "Class 1 exemption dispute. Window condition 6(a) of Schedule 2 PDR 2001 analysed in detail.",
    },
    "r312690a": {
        "title":       "ABP-312690-22 — Rural Housing, Kildare, Urban Influence, Gap Site, Road Safety",
        "filename":    "r312690a.pdf",
        "jurisdiction": Jurisdiction.KILDARE,
        "category":    "RH",
        "year":        2022,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "urban influence", "gap site", "road safety", "Kildare Development Plan"],
        "notes":       "Kildare rural housing policy. Strong urban influence area. Gap site definition analysed.",
    },
    "r314259": {
        "title":       "ABP-314259-22 — Section 5 Referral, Class 1 Extension, 40sqm Threshold, Dublin",
        "filename":    "r314259.pdf",
        "jurisdiction": Jurisdiction.DUBLIN_CITY,
        "category":    "RE",
        "year":        2022,
        "outcome":     "exempted",
        "key_issues":  ["Class 1", "40 square metres", "exempted development", "Section 5", "demolition Class 50"],
        "notes":       "Section 5 referral. Detailed analysis of Class 1 and Class 50(b) exemptions. 40sqm threshold verified.",
    },
    "r314391": {
        "title":       "ABP-314391-22 — Residential Extension, Dormer, Galway City, Residential Amenity",
        "filename":    "r314391.pdf",
        "jurisdiction": Jurisdiction.GALWAY_CITY,
        "category":    "RE",
        "year":        2022,
        "outcome":     "granted",
        "key_issues":  ["dormer extension", "residential amenity", "overshadowing", "overlooking", "garden shed exemption"],
        "notes":       "Dormer bungalow extension. Garden shed exemption condition discussed. Overlooking and overshadowing assessment.",
    },
    "r317659": {
        "title":       "ABP-317659-23 — Section 5 Referral, Class 1 Exceeded, 50sqm Extension Not Exempt, Dublin",
        "filename":    "r317659.pdf",
        "jurisdiction": Jurisdiction.FINGAL,
        "category":    "RE",
        "year":        2023,
        "outcome":     "not_exempted",
        "key_issues":  ["Class 1", "40 square metres", "50 square metres", "not exempted", "Section 5 referral", "Class 50"],
        "notes":       "Key case — 50sqm extension exceeds 40sqm Class 1 threshold. Board confirmed not exempted development.",
    },
    "r318136": {
        "title":       "ABP-318136-23 — Telecommunications Mast, Rural Cork, Protected Structure Proximity",
        "filename":    "r318136.pdf",
        "jurisdiction": Jurisdiction.CORK_COUNTY,
        "category":    "COM",
        "year":        2023,
        "outcome":     "refused",
        "key_issues":  ["telecommunications", "rural area", "SAC proximity", "protected structure", "visual impact"],
        "notes":       "Telecoms mast in rural Cork. SAC proximity and visual impact. Protected structures curtilage.",
    },
    "r318380": {
        "title":       "ABP-318380-23 — Rural Housing, Dublin Glencullen, Strong Urban Pressure, Employment Need",
        "filename":    "r318380.pdf",
        "jurisdiction": Jurisdiction.DUBLIN_CITY,
        "category":    "RH",
        "year":        2023,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "strong urban pressure", "employment need", "locationally specific", "Dublin mountains"],
        "notes":       "Rural housing refused in Dublin Mountains. Applicant failed locationally-specific employment need test.",
    },
    "r319089": {
        "title":       "ABP-319089-24 — Rural Housing, Kildare, Health Circumstances, Urban Influence Refused",
        "filename":    "r319089.pdf",
        "jurisdiction": Jurisdiction.KILDARE,
        "category":    "RH",
        "year":        2024,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "health circumstances", "exceptional circumstances", "urban influence", "Kildare CDP"],
        "notes":       "Health circumstances argument rejected. Strong urban influence area. Section HO O47 Kildare CDP analysed.",
    },
    "r320196": {
        "title":       "ABP-320196-24 — Residential Extension Retention, Fingal, Ground Floor Rear, Refused",
        "filename":    "r320196.pdf",
        "jurisdiction": Jurisdiction.FINGAL,
        "category":    "RE",
        "year":        2024,
        "outcome":     "refused",
        "key_issues":  ["rear extension", "retention", "ground floor extension", "residential amenity", "Fingal CDP Section 14.10"],
        "notes":       "Retention of extension refused. Fingal CDP Section 14.10.2 and 14.10.2.3 analysed in detail.",
    },
    "r320243": {
        "title":       "ABP-320243-24 — Rural Housing, Offaly, Genuine Local Need, Siting and Design",
        "filename":    "r320243.pdf",
        "jurisdiction": Jurisdiction.OFFALY,
        "category":    "RH",
        "year":        2024,
        "outcome":     "refused",
        "key_issues":  ["rural housing", "genuine local need", "siting and design", "Offaly CDP SSP-27", "suburban design"],
        "notes":       "Refused on design grounds. Suburban design not appropriate for rural Offaly. SSP-27 criteria analysed.",
    },
    "r320921": {
        "title":       "ABP-320921-24 — Rural Housing, Mayo, Haphazard Development, Clew Bay SAC",
        "filename":    "r320921.pdf",
        "jurisdiction": Jurisdiction.MAYO,
        "category":    "RH",
        "year":        "2024",
        "outcome":     "refused",
        "key_issues":  ["rural housing", "haphazard development", "random development", "SAC proximity", "Mayo CDP"],
        "notes":       "Refused as random haphazard development. Clew Bay SAC proximity. Mayo rural housing policy.",
    },
    "r321085": {
        "title":       "ABP-321085-24 — Rural Cluster Housing, Kildare, Ribbon Development, Density Exceeded",
        "filename":    "r321085.pdf",
        "jurisdiction": Jurisdiction.KILDARE,
        "category":    "RH",
        "year":        2024,
        "outcome":     "refused",
        "key_issues":  ["cluster housing", "ribbon development", "density", "30 units per square kilometre", "agricultural exemption"],
        "notes":       "5 dwelling cluster refused. 30 units/km2 density threshold. Agricultural occupation exemption analysed.",
    },
    "r321428": {
        "title":       "ABP-321428-24 — Residential Extension, Fingal, Class 1 and Class 3, Garden Room",
        "filename":    "r321428.pdf",
        "jurisdiction": Jurisdiction.FINGAL,
        "category":    "RE",
        "year":        2024,
        "outcome":     "granted",
        "key_issues":  ["Class 1", "Class 3", "garden room", "rear extension", "Fingal CDP", "de-exemption condition"],
        "notes":       "Class 1 and Class 3 exemptions. De-exemption condition on original permission discussed. Garden room assessment.",
    },
    "r321733": {
        "title":       "ABP-321733-25 — Residential Extension, Fingal, ACA, First Floor Rear, Granted",
        "filename":    "r321733.pdf",
        "jurisdiction": Jurisdiction.FINGAL,
        "category":    "RE",
        "year":        2025,
        "outcome":     "granted",
        "key_issues":  ["rear extension", "first floor", "architectural conservation area", "ACA", "residential amenity", "Fingal CDP"],
        "notes":       "First floor rear extension in ACA granted. Fingal CDP Section 14.10.2 and 14.19.3.3 applied.",
    },
    "r322839": {
        "title":       "ACP-322839-25 — Rural Housing Retention, Kildare, Local Need, Occupancy Condition",
        "filename":    "r322839.pdf",
        "jurisdiction": Jurisdiction.KILDARE,
        "category":    "RH",
        "year":        2025,
        "outcome":     "varied",
        "key_issues":  ["rural housing retention", "local need", "occupancy condition", "Kildare CDP HO P11", "retention"],
        "notes":       "Retention of rural dwelling. Local need assessed retrospectively. Occupancy condition implications.",
    },
    "r323011": {
        "title":       "ACP-323011-25 — Apartment Development, Cork County, Design Standards, Parking",
        "filename":    "r323011.pdf",
        "jurisdiction": Jurisdiction.CORK_COUNTY,
        "category":    "APT",
        "year":        2025,
        "outcome":     "granted",
        "key_issues":  ["apartments", "design standards", "parking", "Cork CDP", "accommodation centre", "dual aspect"],
        "notes":       "Apartment development Cork. Design Standards for New Apartments 2023 applied. Parking assessment.",
    },
}


# ── PDF extraction ─────────────────────────────────────────────────────────────

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
        sample    = "".join(reader.pages[i].extract_text() or "" for i in range(min(5, total)))
        avg_chars = len(sample) / min(5, total) if total else 0

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


# ── Ingest one report ─────────────────────────────────────────────────────────

def ingest_report(key: str, r: dict, kb, dry_run: bool) -> dict:
    stats = {
        "key":            key,
        "title":          r["title"][:55],
        "status":         "pending",
        "extraction":     "",
        "raw_chars":      0,
        "chunks_created": 0,
        "chunks_added":   0,
        "errors":         [],
    }

    pdf_path = REPORTS_DIR / r["filename"]

    if not pdf_path.exists():
        stats["status"] = "missing"
        stats["errors"].append(f"File not found: {r['filename']}")
        console.log(f"  [red]✗ Missing:[/] {r['filename']}")
        return stats

    size_kb = pdf_path.stat().st_size / 1024
    console.log(f"\n[cyan]Processing:[/] {r['title'][:70]}")
    console.log(f"  File: {r['filename']} ({size_kb:.0f} KB) | {r['category']} | {r['outcome']}")

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

    # Build enriched source title with outcome and key issues
    enriched_title = (
        f"{r['title']} "
        f"[{r['outcome'].upper()}] "
        f"Key issues: {', '.join(r['key_issues'][:3])}"
    )

    chunker = SemanticChunker(
        document_type  = DocumentType.ABP_DECISION,
        jurisdiction   = r["jurisdiction"],
        source_title   = enriched_title[:200],
        source_url     = f"https://www.pleanala.ie/anbordpleanala/media/abp/cases/reports/{key[1:4]}/{key}.pdf",
        act_year       = int(str(r["year"])[:4]),
        effective_date = None,
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


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run(reports: list[str] = None, dry_run: bool = False) -> list[dict]:

    console.print(Panel.fit(
        "[bold]PlanIQ — Inspector Reports Ingestion[/]\n"
        "18 An Coimisiún Pleanála inspector reports — how Irish planning law is applied in practice",
        border_style="cyan"
    ))

    if not dry_run:
        kb = PlanIQKnowledgeBase()
        console.log(f"[green]✓[/] KB ready — {kb.get_stats()['total_chunks_chroma']:,} existing chunks")
    else:
        kb = None
        console.log("[yellow]DRY RUN — no DB writes[/]")

    selected = {k: v for k, v in REPORTS.items()
                if not reports or k in reports}

    console.log(f"Processing {len(selected)} reports")

    all_stats = []
    for key, r in selected.items():
        result = ingest_report(key, r, kb, dry_run=dry_run)
        all_stats.append(result)

    # Summary
    console.rule("[cyan]Ingestion Complete[/]")
    table = Table(title="Inspector Reports Results", border_style="dim")
    table.add_column("Report",   style="cyan", max_width=12)
    table.add_column("Cat",      width=5)
    table.add_column("Status",   width=12)
    table.add_column("Method",   width=7)
    table.add_column("Chars",    justify="right")
    table.add_column("Chunks",   justify="right")
    table.add_column("Added",    justify="right")

    STATUS = {
        "success":           "[green]success[/]",
        "missing":           "[red]missing[/]",
        "extraction_failed": "[red]failed[/]",
        "no_chunks":         "[yellow]no chunks[/]",
        "dry_run":           "[dim]dry run[/]",
    }

    totals = {"chars": 0, "created": 0, "added": 0}
    for s in all_stats:
        r = REPORTS.get(s["key"], {})
        table.add_row(
            s["key"],
            r.get("category", "-"),
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
        "TOTAL", "", "", "",
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
    parser = argparse.ArgumentParser(description="PlanIQ — Inspector Reports Ingestion")
    parser.add_argument("--report", nargs="+",
                        help=f"Specific reports. Options: {list(REPORTS.keys())}")
    parser.add_argument("--test", action="store_true", help="Dry run")
    args = parser.parse_args()

    run(reports=args.report, dry_run=args.test)
