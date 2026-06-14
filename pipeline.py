"""
PlanIQ — Ingestion Pipeline
PlanIQ — Ingestion Pipeline
Orchestrates the full Step 1 pipeline:

  scraper → chunker → validation → staleness gate → knowledge base

Run this to build or update the knowledge base.
Idempotent — safe to run multiple times (deduplication built in).
"""

import sys
from datetime import date
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent / "knowledge_base"))

from ingestion.scraper  import PlanningDocumentScraper, PLANNING_SOURCES
from ingestion.chunker  import SemanticChunker
from ingestion.schema   import DocumentType, Jurisdiction, ConfidenceLevel, PlanningChunk
from knowledge_base.store import PlanIQKnowledgeBase

console = Console()


def run_ingestion(rebuild: bool = False, sources: list[str] = None) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        rebuild: If True, wipe and rebuild the entire knowledge base
        sources: If set, only ingest these source IDs (default = all)

    Returns:
        Summary statistics dict
    """
    console.print(Panel.fit(
        "[bold]PlanIQ — Step 1: Knowledge Base Ingestion[/]\n"
        "Building the foundation for Irish planning AI",
        border_style="cyan"
    ))

    stats = {
        "sources_attempted": 0,
        "sources_succeeded": 0,
        "total_chars":       0,
        "total_chunks":      0,
        "chunks_added":      0,
        "chunks_rejected_stale": 0,
        "errors":            [],
    }

    # ── Step 1a: Initialise knowledge base ───
    console.rule("[cyan]Step 1a: Knowledge Base[/]")
    kb = PlanIQKnowledgeBase(rebuild=rebuild)

    # ── Step 1b: Scrape documents ─────────────
    console.rule("[cyan]Step 1b: Document Scraping[/]")
    scraper = PlanningDocumentScraper(use_cache=True)

    if sources:
        raw_docs = [scraper.fetch_source(s) for s in sources]
        raw_docs = [d for d in raw_docs if d is not None]
    else:
        raw_docs = scraper.fetch_all()

    stats["sources_attempted"] = len(PLANNING_SOURCES) if not sources else len(sources)
    stats["sources_succeeded"] = len(raw_docs)
    stats["total_chars"]       = sum(d["char_count"] for d in raw_docs)

    console.log(
        f"[green]✓[/] Scraped {stats['sources_succeeded']} documents "
        f"({stats['total_chars']:,} total chars)"
    )

    # ── Step 1c: Chunk each document ──────────
    console.rule("[cyan]Step 1c: Semantic Chunking[/]")
    all_chunks: list[PlanningChunk] = []

    for doc in raw_docs:
        meta = doc["metadata"]
        try:
            chunker = SemanticChunker(
                document_type  = DocumentType(meta["document_type"]),
                jurisdiction   = Jurisdiction(meta["jurisdiction"]),
                source_title   = meta["title"],
                source_url     = meta["url"],
                si_number      = meta.get("si_number", ""),
                act_year       = meta.get("act_year"),
                effective_date = meta.get("effective_date"),
                confidence     = ConfidenceLevel(meta.get("confidence", "medium")),
                is_verbatim    = meta.get("is_verbatim", False),
            )
            chunks = chunker.chunk(doc["raw_text"])
            all_chunks.extend(chunks)
            console.log(f"  [green]✓[/] {meta['id']}: {len(chunks)} chunks")

        except Exception as e:
            err = f"Chunking failed for {meta['id']}: {e}"
            console.log(f"  [red]✗[/] {err}")
            stats["errors"].append(err)

    stats["total_chunks"] = len(all_chunks)
    console.log(f"[green]✓[/] Total chunks created: {stats['total_chunks']}")

    # ── Step 1d: Validation + staleness gate ──
    console.rule("[cyan]Step 1d: Validation + Staleness Gate[/]")
    valid_chunks = []
    stale_count  = 0

    for chunk in all_chunks:
        if chunk.is_stale:
            stale_count += 1
            console.log(
                f"[yellow]⚠ STALE chunk blocked:[/] {chunk.source_title[:40]} "
                f"| superseded_by: {chunk.superseded_by}"
            )
            continue
        valid_chunks.append(chunk)

    stats["chunks_rejected_stale"] = stale_count
    console.log(
        f"[green]✓[/] Validation: {len(valid_chunks)} valid | "
        f"{stale_count} stale rejected"
    )

    # ── Step 1e: Write to knowledge base ──────
    console.rule("[cyan]Step 1e: Writing to Knowledge Base[/]")
    added = kb.add_chunks(valid_chunks)
    stats["chunks_added"] = added

    # ── Summary ────────────────────────────────
    console.rule("[cyan]Ingestion Complete[/]")
    kb_stats = kb.get_stats()

    table = Table(title="Knowledge Base Status", border_style="dim")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Sources ingested",        str(stats["sources_succeeded"]))
    table.add_row("Total raw chars",         f"{stats['total_chars']:,}")
    table.add_row("Chunks created",          str(stats["total_chunks"]))
    table.add_row("Chunks added (new)",      str(stats["chunks_added"]))
    table.add_row("Stale chunks rejected",   str(stats["chunks_rejected_stale"]))
    table.add_row("ChromaDB total",          str(kb_stats["total_chunks_chroma"]))
    table.add_row("BM25 total",              str(kb_stats["total_chunks_bm25"]))
    table.add_row("Needing re-verification", str(kb_stats["chunks_needing_reverify"]))
    table.add_row("Errors",                  str(len(stats["errors"])))

    console.print(table)

    if stats["errors"]:
        console.print("[red]Errors:[/]")
        for e in stats["errors"]:
            console.print(f"  • {e}")

    # Show source breakdown
    sources_table = Table(title="Ingested Sources", border_style="dim")
    sources_table.add_column("Document", style="white", max_width=55)
    sources_table.add_column("Type", style="cyan")
    sources_table.add_column("Jurisdiction", style="green")
    sources_table.add_column("Chunks", justify="right")

    for source in kb.list_sources():
        sources_table.add_row(
            source["title"][:55],
            source["type"],
            source["jurisdiction"],
            str(source["chunk_count"]),
        )
    console.print(sources_table)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlanIQ Knowledge Base Ingestion")
    parser.add_argument("--rebuild",  action="store_true", help="Wipe and rebuild the KB")
    parser.add_argument("--sources",  nargs="+",           help="Specific source IDs to ingest")
    args = parser.parse_args()

    result = run_ingestion(rebuild=args.rebuild, sources=args.sources)
    sys.exit(0 if not result["errors"] else 1)
