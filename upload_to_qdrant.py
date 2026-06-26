"""
PlanIQ — Qdrant Cloud Uploader
================================
Pushes all chunks from local ChromaDB to Qdrant Cloud.

Run ONCE to populate Qdrant Cloud with the full knowledge base.
After this, Streamlit Cloud uses Qdrant instead of local ChromaDB.

Usage:
  python upload_to_qdrant.py --url https://xyz.eu-central.aws.cloud.qdrant.io --key YOUR_API_KEY

The script will:
  1. Connect to your local ChromaDB (13,025 chunks)
  2. Connect to Qdrant Cloud
  3. Create a 'planiq' collection if it doesn't exist
  4. Upload all chunks in batches of 100
  5. Verify the upload count matches
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent / "knowledge_base"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

COLLECTION_NAME = "planiq"
VECTOR_SIZE     = 384  # all-MiniLM-L6-v2 output dimension
BATCH_SIZE      = 100


def upload_to_qdrant(qdrant_url: str, qdrant_api_key: str, dry_run: bool = False):

    console.print(f"\n[cyan]PlanIQ — Qdrant Cloud Uploader[/]")
    console.print(f"Target: {qdrant_url}")
    console.print(f"Collection: {COLLECTION_NAME}")

    # ── Step 1: Load local ChromaDB ───────────────────────────────────────────
    console.print("\n[yellow]Step 1 — Loading local ChromaDB...[/]")
    from knowledge_base.store import PlanIQKnowledgeBase
    kb    = PlanIQKnowledgeBase()
    stats = kb.get_stats()
    total = stats["total_chunks_chroma"]
    console.print(f"  [green]✓[/] Local KB: {total:,} chunks across {stats['total_docs_ingested']} documents")

    # Get all chunks from ChromaDB
    console.print("  Fetching all chunks from ChromaDB...")
    result = kb._collection.get(
        include=["embeddings", "documents", "metadatas"],
        limit=total + 100,
    )

    ids        = result["ids"]
    embeddings = result["embeddings"]
    documents  = result["documents"]
    metadatas  = result["metadatas"]

    console.print(f"  [green]✓[/] Fetched {len(ids):,} chunks with embeddings")

    if dry_run:
        console.print("[dim]DRY RUN — stopping before Qdrant upload[/]")
        return

    # ── Step 2: Connect to Qdrant Cloud ───────────────────────────────────────
    console.print("\n[yellow]Step 2 — Connecting to Qdrant Cloud...[/]")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        OptimizersConfigDiff, HnswConfigDiff
    )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

    # Test connection
    try:
        collections = client.get_collections()
        console.print(f"  [green]✓[/] Connected to Qdrant Cloud")
        existing = [c.name for c in collections.collections]
        console.print(f"  Existing collections: {existing or 'none'}")
    except Exception as e:
        console.print(f"  [red]✗ Connection failed: {e}[/]")
        return

    # ── Step 3: Create collection ─────────────────────────────────────────────
    console.print(f"\n[yellow]Step 3 — Creating collection '{COLLECTION_NAME}'...[/]")
    if COLLECTION_NAME in existing:
        console.print(f"  [yellow]Collection already exists — will overwrite[/]")
        client.delete_collection(COLLECTION_NAME)
        console.print(f"  [green]✓[/] Deleted existing collection")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,  # build HNSW index after upload complete
        ),
    )
    console.print(f"  [green]✓[/] Collection '{COLLECTION_NAME}' created")

    # ── Step 4: Upload in batches ─────────────────────────────────────────────
    console.print(f"\n[yellow]Step 4 — Uploading {len(ids):,} chunks to Qdrant...[/]")
    start     = time.time()
    uploaded  = 0
    failed    = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("  Uploading..."),
        BarColumn(),
        TextColumn("{task.percentage:.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(ids))

        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids   = ids[i:i + BATCH_SIZE]
            batch_embs  = embeddings[i:i + BATCH_SIZE]
            batch_docs  = documents[i:i + BATCH_SIZE]
            batch_metas = metadatas[i:i + BATCH_SIZE]

            points = []
            for j, (chunk_id, emb, doc, meta) in enumerate(
                zip(batch_ids, batch_embs, batch_docs, batch_metas)
            ):
                # Qdrant requires integer or UUID point IDs
                # We use a sequential integer and store the original chunk_id in payload
                point_id = i + j

                payload = {
                    "chunk_id":    chunk_id,
                    "text":        doc,
                    "source_title": meta.get("source_title", ""),
                    "jurisdiction": meta.get("jurisdiction", "national"),
                    "document_type": meta.get("document_type", ""),
                    "section_ref":  meta.get("section_ref", ""),
                    "act_year":     meta.get("act_year", 0),
                    "effective_date": meta.get("effective_date", ""),
                    "confidence":   meta.get("confidence", "high"),
                    "is_stale":     meta.get("is_stale", False),
                    "chunk_index":  meta.get("chunk_index", 0),
                }

                points.append(PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                ))

            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
                uploaded += len(points)
            except Exception as e:
                console.print(f"\n  [red]Batch {i//BATCH_SIZE} failed: {e}[/]")
                failed += len(points)

            progress.advance(task, len(batch_ids))

    elapsed = time.time() - start
    console.print(f"\n  [green]✓[/] Uploaded {uploaded:,} chunks in {elapsed:.1f}s")
    if failed:
        console.print(f"  [red]✗ Failed: {failed} chunks[/]")

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    console.print(f"\n[yellow]Step 5 — Verifying upload...[/]")
    collection_info = client.get_collection(COLLECTION_NAME)
    qdrant_count    = collection_info.points_count

    console.print(f"  Local ChromaDB:  {total:,} chunks")
    console.print(f"  Qdrant Cloud:    {qdrant_count:,} chunks")

    if qdrant_count == total:
        console.print(f"  [green]✓ Upload verified — counts match[/]")
    else:
        console.print(f"  [yellow]⚠ Count mismatch — {total - qdrant_count} chunks missing[/]")

    console.print(f"\n[green]✓ Upload complete![/]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Add these to Streamlit Cloud secrets:")
    console.print(f"     QDRANT_URL = \"{qdrant_url}\"")
    console.print(f"     QDRANT_API_KEY = \"your-key\"")
    console.print(f"     ANTHROPIC_API_KEY = \"your-key\"")
    console.print(f"  2. Deploy streamlit_app.py to Streamlit Cloud")
    console.print(f"  3. Point planiq.ie at the Streamlit Cloud URL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlanIQ — Upload KB to Qdrant Cloud")
    parser.add_argument("--url", required=True, help="Qdrant Cloud cluster URL")
    parser.add_argument("--key", required=True, help="Qdrant Cloud API key")
    parser.add_argument("--test", action="store_true", help="Dry run — no upload")
    args = parser.parse_args()

    upload_to_qdrant(
        qdrant_url=args.url,
        qdrant_api_key=args.key,
        dry_run=args.test,
    )
