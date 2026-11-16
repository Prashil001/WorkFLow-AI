from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select, text
from sqlalchemy.engine import Connection, Engine

from app.models.document_chunk import DocumentChunkRecord
from app.rag.corpus import PRICING_CORPUS
from app.workflow_engine.rag import (
    DEFAULT_MINIMUM_SCORE,
    RetrievedChunk,
    TextEmbedder,
    clamp_retrieval_limit,
    rank_retrieved_chunks,
)


class PgVectorRetriever:
    """Cosine-similarity retrieval over one corpus using the HNSW index."""

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        corpus: str = PRICING_CORPUS,
        bind: Engine | Connection | None = None,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> None:
        self._embedder = embedder
        self._bind = bind
        self.corpus = corpus
        self.minimum_score = minimum_score

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        query_vector = next(iter(self._embedder.query_embed(query)), None)
        if query_vector is None:
            return []

        distance = DocumentChunkRecord.embedding.cosine_distance(
            [float(value) for value in query_vector]
        )
        # Order by the raw distance expression so PostgreSQL can use the
        # HNSW index; thresholding and tie-breaks happen in shared Python code.
        statement = (
            select(
                DocumentChunkRecord.source_id,
                DocumentChunkRecord.client,
                DocumentChunkRecord.title,
                DocumentChunkRecord.chunk_index,
                DocumentChunkRecord.text,
                distance.label("distance"),
            )
            .where(DocumentChunkRecord.corpus == self.corpus)
            .order_by(distance)
            .limit(clamp_retrieval_limit(limit))
        )

        with self._connection() as connection:
            # With a corpus filter, a plain HNSW scan can return fewer rows
            # than requested; iterative scans (pgvector >= 0.8) keep searching
            # in exact distance order. Transaction-local setting.
            connection.execute(text(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', true)"
            ))
            # ef_search bounds the HNSW candidate list; keep it above the
            # requested depth (pgvector's default is 40).
            connection.execute(
                text("SELECT set_config('hnsw.ef_search', :ef, true)"),
                {"ef": str(max(40, 2 * clamp_retrieval_limit(limit)))},
            )
            rows = connection.execute(statement).all()

        return rank_retrieved_chunks(
            (
                RetrievedChunk(
                    source_id=row.source_id,
                    client=row.client,
                    title=row.title,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    score=1.0 - float(row.distance),
                )
                for row in rows
            ),
            minimum_score=self.minimum_score,
            limit=limit,
        )

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        if isinstance(self._bind, Connection):
            yield self._bind
            return

        bind = self._bind
        if bind is None:
            from app.db.database import engine as bind

        with bind.connect() as connection:
            yield connection
