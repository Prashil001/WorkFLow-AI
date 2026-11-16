from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine

from app.models.document_chunk import DocumentChunkRecord
from app.workflow_engine.rag import (
    DEFAULT_MINIMUM_SCORE,
    RetrievedChunk,
    TextEmbedder,
    clamp_retrieval_limit,
    rank_retrieved_chunks,
)


class PgVectorRetriever:
    """Cosine-similarity retrieval over document_chunks using the HNSW index."""

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        bind: Engine | Connection | None = None,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> None:
        self._embedder = embedder
        self._bind = bind
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
            .order_by(distance)
            .limit(clamp_retrieval_limit(limit))
        )

        with self._connection() as connection:
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
