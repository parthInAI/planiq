"""
PlanIQ — Knowledge Base
Manages the dual-index store:
  1. ChromaDB   — dense vector search (semantic similarity)
  2. BM25       — sparse keyword search (exact statute references)

Two indexes, one interface. The retrieval layer fuses both.

Design principles:
  - Stale chunks are NEVER written to either index
  - Every write is logged with timestamp and source metadata
  - The store is queryable by jurisdiction — filters out 30/31 councils instantly
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from schema import PlanningChunk, Jurisdiction, DocumentType

console = Console()

# ── Paths ────────────────────────────────────
KB_DIR       = Path(__file__).parent.parent / "knowledge_base"
CHROMA_DIR   = KB_DIR / "chroma_store"
BM25_PATH    = KB_DIR / "bm25_index.pkl"
MANIFEST_PATH = KB_DIR / "manifest.json"
KB_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding model ───────────────────────────
# all-MiniLM-L6-v2: fast, good for legal-adjacent text, runs locally
# In production: swap for law-ai/legal-bert-base-uncased or fine-tuned model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME  = "planiq_v1"


class PlanIQKnowledgeBase:
    """
    The central knowledge store for PlanIQ.
    Manages ingestion, deduplication, staleness gating, and retrieval prep.
    """

    def __init__(self, rebuild: bool = False):
        console.log("[bold cyan]Initialising PlanIQ Knowledge Base...[/]")

        # ChromaDB — persistent on disk
        self._chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        if rebuild:
            try:
                self._chroma_client.delete_collection(COLLECTION_NAME)
                console.log("[yellow]Rebuilt:[/] deleted existing ChromaDB collection")
            except Exception:
                pass

        self._collection = self._chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}   # cosine distance for planning text
        )

        # Embedding model (runs locally — no API key needed)
        console.log(f"[dim]Loading embedding model: {EMBED_MODEL_NAME}[/]")
        self._embedder = SentenceTransformer(EMBED_MODEL_NAME)

        # BM25 sparse index (in-memory, serialised to disk)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: list[PlanningChunk] = []
        self._load_bm25()

        # Manifest tracks what's been ingested
        self._manifest = self._load_manifest()

        console.log(
            f"[green]✓[/] KB ready — "
            f"ChromaDB: {self._collection.count()} chunks | "
            f"BM25: {len(self._bm25_chunks)} chunks"
        )

    # ── Ingestion ────────────────────────────

    def add_chunks(self, chunks: list[PlanningChunk], overwrite: bool = False) -> int:
        """
        Add chunks to both indexes.
        Stale chunks are silently rejected — never stored.
        Returns count of successfully added chunks.
        """
        valid   = [c for c in chunks if not c.is_stale]
        stale   = [c for c in chunks if c.is_stale]

        if stale:
            console.log(f"[yellow]⚠ Rejected {len(stale)} stale chunks[/]")

        if not valid:
            console.log("[red]✗ No valid chunks to add[/]")
            return 0

        # Deduplication — skip chunks already in ChromaDB
        if not overwrite:
            existing_ids = set(self._collection.get()["ids"])
            new_chunks   = [c for c in valid if c.chunk_id not in existing_ids]
            if len(new_chunks) < len(valid):
                console.log(f"[dim]Skipped {len(valid) - len(new_chunks)} duplicate chunks[/]")
            valid = new_chunks

        if not valid:
            return 0

        # ── Write to ChromaDB ─────────────────
        texts      = [c.text for c in valid]
        embeddings = self._embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        self._collection.add(
            ids        = [c.chunk_id for c in valid],
            documents  = texts,
            embeddings = embeddings,
            metadatas  = [c.to_chroma_metadata() for c in valid],
        )

        # ── Update BM25 ──────────────────────
        self._bm25_chunks.extend(valid)
        self._rebuild_bm25()
        self._save_bm25()

        # ── Update manifest ──────────────────
        for chunk in valid:
            self._manifest["ingested_docs"].setdefault(chunk.source_doc_id, {
                "title":       chunk.source_title,
                "type":        chunk.document_type.value,
                "jurisdiction": chunk.jurisdiction.value,
                "chunk_count": 0,
                "first_seen":  datetime.now().isoformat(),
            })
            self._manifest["ingested_docs"][chunk.source_doc_id]["chunk_count"] += 1
            self._manifest["last_updated"] = datetime.now().isoformat()

        self._save_manifest()

        console.log(f"[green]✓[/] Added {len(valid)} chunks to both indexes")
        return len(valid)

    # ── Retrieval (called by the retrieval layer) ──

    def dense_search(
        self,
        query: str,
        n_results: int = 20,
        jurisdiction_filter: Optional[Jurisdiction] = None,
        doc_type_filter: Optional[DocumentType] = None,
    ) -> list[dict]:
        """
        Semantic vector search via ChromaDB.
        Applies jurisdiction pre-filter — eliminates irrelevant councils.
        Returns raw ChromaDB result dicts (reranker processes these).
        """
        query_embedding = self._embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        # Build metadata filter — all conditions must use $and
        if jurisdiction_filter and doc_type_filter:
            where = {"$and": [
                {"is_stale": False},
                {"document_type": doc_type_filter.value},
                {"$or": [
                    {"jurisdiction": jurisdiction_filter.value},
                    {"jurisdiction": "national"},
                ]}
            ]}
        elif jurisdiction_filter:
            where = {"$and": [
                {"is_stale": False},
                {"$or": [
                    {"jurisdiction": jurisdiction_filter.value},
                    {"jurisdiction": "national"},
                ]}
            ]}
        elif doc_type_filter:
            where = {"$and": [
                {"is_stale": False},
                {"document_type": doc_type_filter.value},
            ]}
        else:
            where = {"is_stale": False}

        results = self._collection.query(
            query_embeddings = query_embedding,
            n_results        = min(n_results, max(1, self._collection.count())),
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB response into list of dicts
        flat = []
        for i, doc in enumerate(results["documents"][0]):
            flat.append({
                "text":     doc,
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # cosine similarity
                "source":   "dense",
            })
        return flat

    def sparse_search(
        self,
        query: str,
        n_results: int = 20,
        jurisdiction_filter: Optional[Jurisdiction] = None,
    ) -> list[dict]:
        """
        BM25 keyword search — critical for exact statute references.
        "Class 1 Schedule 2" or "Section 5 declaration" must hit exactly.
        """
        if not self._bm25 or not self._bm25_chunks:
            return []

        tokens  = self._tokenise(query)
        scores  = self._bm25.get_scores(tokens)

        # Get top N by score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results * 2]  # Over-fetch then filter

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            chunk = self._bm25_chunks[idx]
            if chunk.is_stale:
                continue
            if jurisdiction_filter and chunk.jurisdiction not in (
                jurisdiction_filter, Jurisdiction.NATIONAL
            ):
                continue
            results.append({
                "text":     chunk.text,
                "metadata": chunk.to_chroma_metadata(),
                "score":    float(scores[idx]),
                "source":   "sparse",
                "chunk":    chunk,
            })
            if len(results) >= n_results:
                break

        return results

    # ── Status & maintenance ─────────────────

    def get_stats(self) -> dict:
        """Return current knowledge base statistics."""
        manifest = self._manifest
        return {
            "total_chunks_chroma":    self._collection.count(),
            "total_chunks_bm25":      len(self._bm25_chunks),
            "total_docs_ingested":    len(manifest.get("ingested_docs", {})),
            "last_updated":           manifest.get("last_updated", "never"),
            "stale_chunks_bm25":      sum(1 for c in self._bm25_chunks if c.is_stale),
            "chunks_needing_reverify": sum(1 for c in self._bm25_chunks if c.needs_reverification),
        }

    def list_sources(self) -> list[dict]:
        """List all ingested source documents."""
        return [
            {"doc_id": doc_id, **info}
            for doc_id, info in self._manifest.get("ingested_docs", {}).items()
        ]

    # ── Private ──────────────────────────────

    def _rebuild_bm25(self):
        """Rebuild BM25 index from current chunk list."""
        tokenised   = [self._tokenise(c.text) for c in self._bm25_chunks if not c.is_stale]
        self._bm25  = BM25Okapi(tokenised)

    def _tokenise(self, text: str) -> list[str]:
        """
        Tokenise text for BM25.
        Preserves legal tokens: "S.I.", "No.", "section", "Class", section numbers.
        """
        import re
        # Lowercase but preserve key legal abbreviations
        text   = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text)
        return tokens

    def _save_bm25(self):
        with open(BM25_PATH, "wb") as f:
            pickle.dump((self._bm25, self._bm25_chunks), f)

    def _load_bm25(self):
        if BM25_PATH.exists():
            with open(BM25_PATH, "rb") as f:
                self._bm25, self._bm25_chunks = pickle.load(f)
            console.log(f"[dim]BM25 loaded from disk: {len(self._bm25_chunks)} chunks[/]")

    def _load_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            return json.loads(MANIFEST_PATH.read_text())
        return {"ingested_docs": {}, "last_updated": None, "schema_version": "1.0"}

    def _save_manifest(self):
        MANIFEST_PATH.write_text(json.dumps(self._manifest, indent=2))
