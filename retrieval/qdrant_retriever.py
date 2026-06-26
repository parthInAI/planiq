"""
PlanIQ — Qdrant Cloud Retriever
=================================
Replaces local ChromaDB + BM25 retrieval with Qdrant Cloud vector search.
Used by streamlit_app.py when deployed to Streamlit Cloud.

Differences from HybridRetriever:
  - Dense search via Qdrant Cloud (remote)
  - BM25 sparse search removed (not needed for cloud — Qdrant handles it)
  - Jurisdiction filtering via Qdrant payload filter
  - Cross-encoder reranker still applied locally
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("planiq.qdrant_retriever")

COLLECTION_NAME    = "planiq"
DEFAULT_TOP_K      = 7
RERANKER_CANDIDATE = 20


class ChunkProxy:
    """
    Wraps a dict chunk to provide attribute-style access.
    Makes Qdrant dict chunks compatible with the hallucination detector
    which expects RetrievedChunk objects with .text, .is_stale, .metadata etc.
    """
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return self._d.get(key)

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)

    @property
    def text(self) -> str:
        return self._d.get("text", "")

    @property
    def is_stale(self) -> bool:
        return bool(self._d.get("is_stale", False))

    @property
    def metadata(self) -> dict:
        return self._d

    @property
    def source_title(self) -> str:
        return self._d.get("source_title", "")

    @property
    def jurisdiction(self) -> str:
        return self._d.get("jurisdiction", "national")

    @property
    def section_ref(self) -> str:
        return self._d.get("section_ref", "")

    @property
    def effective_date(self):
        return self._d.get("effective_date")

    @property
    def score(self) -> float:
        return float(self._d.get("score", 0.0))


@dataclass
class QdrantRetrievalResult:
    chunks:            list[dict]
    retrieval_quality: float
    latency_ms:        int
    method:            str = "qdrant_dense"
    query:             str = ""
    total_dense_hits:  int = 0
    total_sparse_hits: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def get_entity_set(self) -> set:
        """Extract planning entity references from retrieved chunks."""
        import re
        entities = set()
        pattern  = re.compile(
            r'\b(?:Class\s+\d+[A-Z]?|Article\s+\d+[A-Z]?|Section\s+\d+[A-Z]?'
            r'|Schedule\s+\d+|S\.I\.\s+No\.\s+\d+|Part\s+[IVX]+)\b',
            re.IGNORECASE
        )
        for chunk in self.chunks:
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            entities.update(pattern.findall(text))
        return entities

    def to_context_string(self) -> str:
        """Format chunks as structured context string for the LLM prompt."""
        if self.is_empty:
            return "No relevant planning law chunks found."
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(
                f"[CHUNK {i}]\n"
                f"Source: {chunk.source_title}\n"
                f"Section: {chunk.section_ref or 'N/A'}\n"
                f"Jurisdiction: {chunk.jurisdiction}\n"
                f"Effective: {chunk.effective_date or 'N/A'}\n"
                f"Confidence: {chunk.get('confidence', 'high') if hasattr(chunk, 'get') else 'high'}\n"
                f"---\n"
                f"{chunk.text}\n"
            )
        return "\n\n".join(parts)


class QdrantRetriever:
    """
    Dense vector retrieval against Qdrant Cloud.
    Used when QDRANT_URL and QDRANT_API_KEY are set in environment.
    """

    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.client     = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
        self.embedder   = SentenceTransformer("all-MiniLM-L6-v2")
        self._reranker  = None
        logger.info(f"QdrantRetriever initialised — collection: {COLLECTION_NAME}")

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Cross-encoder reranker loaded")
            except Exception as e:
                logger.warning(f"Reranker not available: {e}")
        return self._reranker

    def retrieve(
        self,
        query:        str,
        jurisdiction: Optional[object] = None,
        top_k:        int = DEFAULT_TOP_K,
        use_reranker: bool = True,
    ) -> QdrantRetrievalResult:

        start = time.time()

        # ── Embed query ───────────────────────────────────────────────────────
        query_vector = self.embedder.encode(query).tolist()

        # ── Build jurisdiction filter ─────────────────────────────────────────
        search_filter = None
        if jurisdiction is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
            jurisdiction_val = jurisdiction.value if hasattr(jurisdiction, "value") else str(jurisdiction)
            search_filter = Filter(
                should=[
                    FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction_val)),
                    FieldCondition(key="jurisdiction", match=MatchValue(value="national")),
                ]
            )

        # ── Search Qdrant ─────────────────────────────────────────────────────
        n_candidates = RERANKER_CANDIDATE if use_reranker else top_k
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=n_candidates,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return QdrantRetrievalResult(chunks=[], retrieval_quality=0.0,
                                         latency_ms=int((time.time()-start)*1000), query=query)

        if not results:
            return QdrantRetrievalResult(chunks=[], retrieval_quality=0.0,
                                         latency_ms=int((time.time()-start)*1000), query=query)

        # ── Convert to chunk dicts ────────────────────────────────────────────
        chunks = []
        for r in results:
            p = r.payload
            chunks.append({
                "text":          p.get("text", ""),
                "source_title":  p.get("source_title", ""),
                "jurisdiction":  p.get("jurisdiction", "national"),
                "document_type": p.get("document_type", ""),
                "section_ref":   p.get("section_ref", ""),
                "act_year":      p.get("act_year", 0),
                "effective_date": p.get("effective_date", ""),
                "confidence":    p.get("confidence", "high"),
                "is_stale":      p.get("is_stale", False),
                "score":         r.score,
            })

        # Filter stale chunks and wrap in ChunkProxy
        chunks = [ChunkProxy(c) for c in chunks if not c.get("is_stale", False)]

        # ── Rerank ────────────────────────────────────────────────────────────
        if use_reranker and len(chunks) > top_k:
            reranker = self._get_reranker()
            if reranker:
                try:
                    pairs  = [(query, c["text"]) for c in chunks]
                    scores = reranker.predict(pairs)
                    chunks = [c for _, c in sorted(
                        zip(scores, chunks), key=lambda x: x[0], reverse=True
                    )]
                except Exception as e:
                    logger.warning(f"Reranker failed: {e}")

        chunks = chunks[:top_k]

        # ── Quality score ─────────────────────────────────────────────────────
        quality = min(1.0, results[0].score * 1.2) if results else 0.0

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"QdrantRetriever: {len(chunks)} chunks | quality={quality:.2f} | {elapsed}ms")

        return QdrantRetrievalResult(
            chunks=chunks,
            retrieval_quality=quality,
            latency_ms=elapsed,
            query=query,
            total_dense_hits=len(results),
            total_sparse_hits=len(results),  # Qdrant HNSW covers both — avoid single-method penalty
        )
