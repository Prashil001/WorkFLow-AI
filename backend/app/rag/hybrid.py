"""Lexical retrieval, Reciprocal Rank Fusion, and cross-encoder reranking.

All three implement the shared ``Retriever`` protocol, so they can be composed:
``RerankingRetriever(HybridRetriever(PgVectorRetriever(...), PgFullTextRetriever(...)))``.
"""

from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.rag.corpus import DOCS_CORPUS
from app.rag.pgvector_retriever import connect
from app.workflow_engine.rag import (
    MAX_RETRIEVAL_LIMIT,
    RetrievedChunk,
    Retriever,
    clamp_retrieval_limit,
)


RRF_K = 60  # constant from Cormack et al. (2009); not tuned on the eval set
RERANK_DEPTH = 20

# plainto_tsquery safely parses free text (stemming, stop words, quoting) into an
# AND query; swapping the operators turns it into an OR over the query's terms,
# since natural-language questions rarely have every term in one chunk.
_FULL_TEXT_SQL = text("""
WITH q AS (
    SELECT replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')::tsquery AS query
)
SELECT c.source_id, c.client, c.title, c.chunk_index, c.text,
       ts_rank_cd(c.search_vector, q.query) AS rank
FROM document_chunks AS c, q
WHERE c.corpus = :corpus AND c.search_vector @@ q.query
ORDER BY rank DESC, c.source_id, c.chunk_index
LIMIT :limit
""")


class PgFullTextRetriever:
    """Postgres full-text search over the generated tsvector (GIN index).

    Ranking is ``ts_rank_cd`` (cover density: term frequency and proximity).
    It has no inverse document frequency, so it is lexical but not BM25.
    """

    def __init__(
        self,
        *,
        corpus: str = DOCS_CORPUS,
        bind: Engine | Connection | None = None,
    ) -> None:
        self.corpus = corpus
        self._bind = bind

    def retrieve(self, query: str, *, limit: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        with connect(self._bind) as connection:
            rows = connection.execute(
                _FULL_TEXT_SQL,
                {"query": query, "corpus": self.corpus, "limit": clamp_retrieval_limit(limit)},
            ).all()
        return [
            RetrievedChunk(
                source_id=row.source_id,
                client=row.client,
                title=row.title,
                chunk_index=row.chunk_index,
                text=row.text,
                score=float(row.rank),
            )
            for row in rows
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked lists: score(chunk) = sum over lists of 1 / (k + rank)."""
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = (chunk.source_id, chunk.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, chunk)

    ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    return [_with_score(chunks[key], scores[key]) for key in ordered]


class HybridRetriever:
    """Dense + lexical retrieval merged with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        candidates: int = MAX_RETRIEVAL_LIMIT,
        k: int = RRF_K,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidates = candidates
        self.k = k

    def retrieve(self, query: str, *, limit: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        fused = reciprocal_rank_fusion(
            [
                self.dense.retrieve(query, limit=self.candidates),
                self.lexical.retrieve(query, limit=self.candidates),
            ],
            k=self.k,
        )
        return fused[: clamp_retrieval_limit(limit)]


class PairScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedCrossEncoder:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(value) for value in self._model.rerank(query, list(documents))]


class RerankingRetriever:
    """Rescore the base retriever's top ``depth`` chunks with a cross-encoder.

    Chunks below ``depth`` keep their base order after the reranked ones, so
    callers asking for more than ``depth`` results still get a full list.
    """

    def __init__(
        self,
        base: Retriever,
        scorer: PairScorer,
        *,
        depth: int = RERANK_DEPTH,
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.depth = depth

    def retrieve(self, query: str, *, limit: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        candidates = self.base.retrieve(query, limit=max(limit, self.depth))
        head, tail = candidates[: self.depth], candidates[self.depth :]
        if not head:
            return []
        scores = self.scorer.score(query, [chunk.text for chunk in head])
        if len(scores) != len(head):
            raise ValueError("Cross-encoder returned an unexpected number of scores")
        reranked = [
            _with_score(chunk, score)
            for chunk, score in sorted(
                zip(head, scores),
                key=lambda pair: (-pair[1], pair[0].source_id, pair[0].chunk_index),
            )
        ]
        return (reranked + tail)[: clamp_retrieval_limit(limit)]


def _with_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=chunk.source_id,
        client=chunk.client,
        title=chunk.title,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        score=score,
    )
