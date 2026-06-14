"""
PlanIQ — Hybrid Retrieval Engine (Step 2)
=========================================
Takes a user query and returns the most relevant planning law chunks
using three-stage retrieval:

  Stage 1A — BM25 sparse search    (exact statute keywords)
  Stage 1B — Dense vector search   (semantic similarity)
  Stage 2  — Reciprocal Rank Fusion (merge + deduplicate both lists)
  Stage 3  — Cross-encoder reranker (precision scoring on top-N)
  Stage 4  — Metadata staleness gate (hard block stale chunks)

Why three stages?
  BM25 alone misses paraphrased queries ("can I build an extension?"
  won't hit "Class 1 exempted development rear extension").
  Dense alone misses exact references ("Section 5 declaration" must
  hit exactly — semantic similarity is not enough for statute numbers).
  The reranker fixes both — it reads query + chunk together and scores
  genuine relevance, not just keyword overlap or vector proximity.

Output: RetrievalResult — a ranked list of PlanningChunk objects with
        scores, sources, and a confidence signal for the hallucination layer.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_base"))

from ingestion.schema import Jurisdiction, DocumentType
from knowledge_base.store import PlanIQKnowledgeBase

console = Console()

# ── Retrieval config ─────────────────────────────
DENSE_FETCH       = 30   # chunks fetched from ChromaDB
SPARSE_FETCH      = 30   # chunks fetched from BM25
RRF_K             = 60   # Reciprocal Rank Fusion constant (standard = 60)
RERANK_TOP_N      = 20   # candidates fed to cross-encoder
FINAL_TOP_K       = 7    # chunks returned to generation layer
MIN_SCORE         = 0.01 # minimum RRF score to include a chunk


@dataclass
class RetrievedChunk:
    """A single chunk with its retrieval scores and provenance."""
    text:         str
    metadata:     dict
    rrf_score:    float = 0.0
    rerank_score: float = 0.0
    dense_rank:   Optional[int] = None
    sparse_rank:  Optional[int] = None
    source:       str = ""   # "dense", "sparse", or "both"

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    @property
    def section_ref(self) -> str:
        return self.metadata.get("section_ref", "")

    @property
    def source_title(self) -> str:
        return self.metadata.get("source_title", "")

    @property
    def jurisdiction(self) -> str:
        return self.metadata.get("jurisdiction", "national")

    @property
    def document_type(self) -> str:
        return self.metadata.get("document_type", "")

    @property
    def confidence(self) -> str:
        return self.metadata.get("confidence", "medium")

    @property
    def is_stale(self) -> bool:
        return self.metadata.get("is_stale", False)

    @property
    def effective_date(self) -> str:
        return self.metadata.get("effective_date", "")


@dataclass
class RetrievalResult:
    """
    Complete retrieval result returned to the generation layer.
    Contains ranked chunks + diagnostic signals for hallucination detection.
    """
    query:              str
    chunks:             list[RetrievedChunk] = field(default_factory=list)
    jurisdiction_used:  Optional[str] = None
    total_dense_hits:   int = 0
    total_sparse_hits:  int = 0
    reranker_used:      bool = False
    retrieval_quality:  float = 0.0  # 0-1 signal for confidence scoring

    @property
    def top_chunk(self) -> Optional[RetrievedChunk]:
        return self.chunks[0] if self.chunks else None

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def to_context_string(self) -> str:
        """
        Format chunks as a structured context string for the LLM prompt.
        Each chunk is labelled with its source and section reference —
        the LLM must cite these in its response (hallucination defence).
        """
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
                f"Confidence: {chunk.confidence}\n"
                f"---\n"
                f"{chunk.text}\n"
            )
        return "\n\n".join(parts)

    def get_entity_set(self) -> set[str]:
        """
        Extract all verifiable entities from retrieved chunks.
        Fed to the HalluGraph entity grounding checker.
        Entities = section refs, jurisdiction names, numeric thresholds.
        """
        import re
        entities = set()
        for chunk in self.chunks:
            # Section references
            if chunk.section_ref:
                for ref in chunk.section_ref.split("|"):
                    entities.add(ref.strip().lower())
            # Jurisdiction
            entities.add(chunk.jurisdiction.lower())
            # Numeric thresholds (e.g. "40 square metres", "4 metres")
            numbers = re.findall(r'\d+(?:\.\d+)?\s*(?:square metres?|metres?|sq\.?\s*m)', chunk.text, re.I)
            entities.update(n.lower().strip() for n in numbers)
            # S.I. numbers
            si_refs = re.findall(r'S\.I\.\s*No\.\s*\d+\s*of\s*\d{4}', chunk.text)
            entities.update(s.lower() for s in si_refs)
        return entities


class HybridRetriever:
    """
    Three-stage hybrid retriever for Irish planning law queries.
    Combines BM25 + dense vector search via Reciprocal Rank Fusion,
    then reranks with a cross-encoder for precision.
    """

    def __init__(self, kb: PlanIQKnowledgeBase):
        self.kb        = kb
        self._reranker = None  # lazy-loaded on first use
        console.log("[dim]HybridRetriever initialised[/]")

    def retrieve(
        self,
        query:          str,
        jurisdiction:   Optional[Jurisdiction] = None,
        doc_type:       Optional[DocumentType] = None,
        top_k:          int = FINAL_TOP_K,
        use_reranker:   bool = True,
    ) -> RetrievalResult:
        """
        Main retrieval entry point.

        Args:
            query:        Natural language planning question
            jurisdiction: Filter to specific council (+ national always included)
            doc_type:     Filter to specific document type
            top_k:        Number of chunks to return (default 5)
            use_reranker: Whether to apply cross-encoder reranking

        Returns:
            RetrievalResult with ranked chunks and diagnostic metadata
        """
        console.log(f"[cyan]Retrieving:[/] {query[:70]}...")

        result = RetrievalResult(
            query=query,
            jurisdiction_used=jurisdiction.value if jurisdiction else "all",
        )

        # ── Stage 1A: Dense vector search ────────
        dense_hits = self.kb.dense_search(
            query=query,
            n_results=DENSE_FETCH,
            jurisdiction_filter=jurisdiction,
            doc_type_filter=doc_type,
        )
        result.total_dense_hits = len(dense_hits)

        # ── Stage 1B: BM25 sparse search ─────────
        sparse_hits = self.kb.sparse_search(
            query=query,
            n_results=SPARSE_FETCH,
            jurisdiction_filter=jurisdiction,
        )
        result.total_sparse_hits = len(sparse_hits)

        # ── Stage 1C: Exemption schedule boost ────
        # Always inject top Schedule 2 chunks for extension/exemption queries
        # so Class 1/2/3 content is never missed due to semantic distance
        _exemption_keywords = [
            "extension", "shed", "solar", "exempt", "garage", "porch",
            "fence", "attic", "class 1", "class 2", "schedule 2", "permission"
        ]
        if any(kw in query.lower() for kw in _exemption_keywords):
            from ingestion.schema import DocumentType as _DT
            boost_hits = self.kb.dense_search(
                query="exempted development class schedule dwellinghouse extension floor area",
                n_results=10,
                doc_type_filter=_DT.EXEMPTION_SCHEDULE,
            )
            # Add boost hits to dense pool — RRF will rerank them
            dense_hits = dense_hits + [h for h in boost_hits
                                       if h["metadata"].get("chunk_id") not in
                                       {d["metadata"].get("chunk_id") for d in dense_hits}]

        console.log(
            f"[dim]  Dense: {len(dense_hits)} hits | "
            f"Sparse: {len(sparse_hits)} hits[/]"
        )

        # ── Stage 2: Reciprocal Rank Fusion ───────
        fused = self._reciprocal_rank_fusion(dense_hits, sparse_hits)

        # Hard filter: remove stale chunks
        fused = [c for c in fused if not c.is_stale]

        if not fused:
            console.log("[yellow]⚠ No valid chunks after staleness filter[/]")
            return result

        # ── Stage 3: Cross-encoder reranking ──────
        candidates = fused[:RERANK_TOP_N]

        if use_reranker and len(candidates) > 1:
            candidates    = self._rerank(query, candidates)
            result.reranker_used = True

        # ── Final selection ───────────────────────
        result.chunks = candidates[:top_k]

        # Compute retrieval quality signal (0-1)
        result.retrieval_quality = self._compute_quality(result)

        console.log(
            f"[green]✓[/] Retrieved {len(result.chunks)} chunks "
            f"| quality={result.retrieval_quality:.2f} "
            f"| reranked={result.reranker_used}"
        )

        return result

    # ── Private: RRF fusion ───────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_hits:  list[dict],
        sparse_hits: list[dict],
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion — merges dense and sparse ranked lists.
        RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)

        Why RRF over score normalisation?
          Scores from BM25 and cosine similarity are not comparable scales.
          RRF uses only rank positions — robust and parameter-free.
        """
        scores:   dict[str, float] = {}
        chunks:   dict[str, RetrievedChunk] = {}
        d_ranks:  dict[str, int] = {}
        s_ranks:  dict[str, int] = {}

        # Score dense hits
        for rank, hit in enumerate(dense_hits, 1):
            cid = hit["metadata"].get("chunk_id", f"dense_{rank}")
            scores[cid]  = scores.get(cid, 0) + 1 / (RRF_K + rank)
            d_ranks[cid] = rank
            if cid not in chunks:
                chunks[cid] = RetrievedChunk(
                    text=hit["text"],
                    metadata=hit["metadata"],
                    source="dense",
                )

        # Score sparse hits
        for rank, hit in enumerate(sparse_hits, 1):
            cid = hit["metadata"].get("chunk_id", f"sparse_{rank}")
            scores[cid]  = scores.get(cid, 0) + 1 / (RRF_K + rank)
            s_ranks[cid] = rank
            if cid not in chunks:
                chunks[cid] = RetrievedChunk(
                    text=hit["text"],
                    metadata=hit["metadata"],
                    source="sparse",
                )
            else:
                chunks[cid].source = "both"  # appeared in both lists

        # Annotate chunks with rank info and RRF score
        for cid, chunk in chunks.items():
            chunk.rrf_score  = scores.get(cid, 0)
            chunk.dense_rank = d_ranks.get(cid)
            chunk.sparse_rank = s_ranks.get(cid)

        # Sort by RRF score descending
        fused = sorted(chunks.values(), key=lambda c: c.rrf_score, reverse=True)

        # Filter out very low scores
        fused = [c for c in fused if c.rrf_score >= MIN_SCORE]

        return fused

    # ── Private: Cross-encoder reranker ──────────

    def _rerank(
        self,
        query:      str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Cross-encoder reranking — reads query + chunk text together
        and assigns a genuine relevance score.

        Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        Runs locally, ~50ms per candidate on CPU.

        Why cross-encoder over bi-encoder for reranking?
          Bi-encoders embed query and chunk independently — they miss
          interaction between query terms and chunk content.
          Cross-encoders read both together — much higher precision,
          but too slow for full retrieval so we use it only on top-20.
        """
        reranker = self._get_reranker()
        if reranker is None:
            console.log("[yellow]⚠ Reranker unavailable — returning RRF order[/]")
            return candidates

        try:
            pairs  = [(query, c.text[:512]) for c in candidates]
            scores = reranker.predict(pairs)

            for chunk, score in zip(candidates, scores):
                chunk.rerank_score = float(score)

            reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
            return reranked

        except Exception as e:
            console.log(f"[yellow]⚠ Reranker error: {e} — returning RRF order[/]")
            return candidates

    def _get_reranker(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
            console.log("[dim]Loading cross-encoder reranker...[/]")
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            console.log("[green]✓[/] Cross-encoder loaded")
            return self._reranker
        except Exception as e:
            console.log(f"[yellow]⚠ Could not load cross-encoder: {e}[/]")
            return None

    # ── Private: Quality scoring ──────────────────

    def _compute_quality(self, result: RetrievalResult) -> float:
        """
        Compute a 0-1 retrieval quality score.
        Fed to the hallucination detection layer as a confidence signal.

        Factors:
          - Did both dense AND sparse find results? (+0.3)
          - Did any chunk appear in both lists? (+0.2)
          - Top rerank score (if available)
          - Number of chunks retrieved vs requested
        """
        if result.is_empty:
            return 0.0

        score = 0.0

        # Both retrieval methods returned results
        if result.total_dense_hits > 0 and result.total_sparse_hits > 0:
            score += 0.3

        # Any chunk appeared in both dense and sparse (strong signal)
        both_count = sum(1 for c in result.chunks if c.source == "both")
        if both_count > 0:
            score += 0.2 * min(both_count / len(result.chunks), 1.0)

        # Reranker confidence (normalised sigmoid of top rerank score)
        if result.reranker_used and result.chunks:
            import math
            top_score = result.chunks[0].rerank_score
            # Sigmoid normalisation — reranker scores are logits
            sigmoid   = 1 / (1 + math.exp(-top_score / 5))
            score    += 0.4 * sigmoid

        # Coverage — did we fill all requested slots?
        score += 0.1 * (len(result.chunks) / FINAL_TOP_K)

        return min(score, 1.0)

    # ── Diagnostics ──────────────────────────────

    def explain(self, result: RetrievalResult) -> None:
        """Print a diagnostic table of retrieved chunks."""
        table = Table(
            title=f"Retrieval: '{result.query[:50]}'",
            border_style="dim"
        )
        table.add_column("Rank",     width=5)
        table.add_column("Source",   width=8)
        table.add_column("RRF",      width=7)
        table.add_column("Rerank",   width=8)
        table.add_column("Section",  width=25)
        table.add_column("Jurisdiction", width=14)
        table.add_column("Text preview", width=45)

        for i, chunk in enumerate(result.chunks, 1):
            table.add_row(
                str(i),
                chunk.source,
                f"{chunk.rrf_score:.4f}",
                f"{chunk.rerank_score:.2f}" if result.reranker_used else "—",
                chunk.section_ref[:25] or "—",
                chunk.jurisdiction,
                chunk.text[:45].replace("\n", " ") + "...",
            )

        console.print(table)
        console.print(
            f"Quality: [bold]{result.retrieval_quality:.2f}[/] | "
            f"Dense hits: {result.total_dense_hits} | "
            f"Sparse hits: {result.total_sparse_hits} | "
            f"Jurisdiction: {result.jurisdiction_used}"
        )
