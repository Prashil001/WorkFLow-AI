from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any, Protocol

from app.workflow_engine.mock_data import load_mock_business_data


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 22
DEFAULT_CHUNK_OVERLAP = 5
DEFAULT_MINIMUM_SCORE = 0.70
MAX_RETRIEVAL_LIMIT = 10


class TextEmbedder(Protocol):
    def embed(self, documents: Iterable[str]) -> Iterable[Sequence[float]]: ...

    def query_embed(self, query: str) -> Iterable[Sequence[float]]: ...


@dataclass(frozen=True)
class DocumentChunk:
    source_id: str
    client: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    client: str
    title: str
    chunk_index: int
    text: str
    score: float


def chunk_documents(
    documents: list[dict[str, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than it")

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        words = document["content"].split()
        for chunk_index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break

            chunks.append(DocumentChunk(
                source_id=document["document_id"],
                client=document["client"],
                title=document["title"],
                chunk_index=chunk_index,
                text=(
                    f"{document['title']} for {document['client']}: "
                    f"{' '.join(chunk_words)}"
                ),
            ))
            if start + chunk_size >= len(words):
                break

    return chunks


class Retriever(Protocol):
    """Ranked chunk retrieval shared by the in-memory and pgvector backends."""

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[RetrievedChunk]: ...


def clamp_retrieval_limit(limit: int) -> int:
    return max(1, min(limit, MAX_RETRIEVAL_LIMIT))


def rank_retrieved_chunks(
    chunks: Iterable[RetrievedChunk],
    *,
    minimum_score: float,
    limit: int,
) -> list[RetrievedChunk]:
    """Apply the shared score threshold, tie-break order, and result limit."""
    eligible = [chunk for chunk in chunks if chunk.score >= minimum_score]
    eligible.sort(
        key=lambda chunk: (-chunk.score, chunk.source_id, chunk.chunk_index)
    )
    return eligible[: clamp_retrieval_limit(limit)]


class FastEmbedTextEmbedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    def embed(self, documents: Iterable[str]) -> Iterable[Sequence[float]]:
        return self._model.embed(list(documents))

    def query_embed(self, query: str) -> Iterable[Sequence[float]]:
        return self._model.query_embed(query)


class InMemoryRetriever:
    def __init__(
        self,
        documents: list[dict[str, str]],
        embedder: TextEmbedder,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> None:
        self.chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self.minimum_score = minimum_score
        self._vectors = list(
            embedder.embed([chunk.text for chunk in self.chunks])
        )
        if len(self._vectors) != len(self.chunks):
            raise ValueError("Embedder returned an unexpected number of vectors")
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip() or not self.chunks:
            return []

        query_vector = next(iter(self._embedder.query_embed(query)), None)
        if query_vector is None:
            return []

        return rank_retrieved_chunks(
            (
                RetrievedChunk(
                    source_id=chunk.source_id,
                    client=chunk.client,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=_cosine_similarity(query_vector, vector),
                )
                for chunk, vector in zip(self.chunks, self._vectors, strict=True)
            ),
            minimum_score=self.minimum_score,
            limit=limit,
        )


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have matching dimensions")

    dot_product = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sqrt(sum(float(value) ** 2 for value in left))
    right_norm = sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@lru_cache(maxsize=1)
def get_default_retriever() -> Retriever:
    """Build the retriever selected by RAG_RETRIEVER (memory or pgvector)."""
    from app.core.config import get_settings

    backend = get_settings().rag_retriever
    if backend == "pgvector":
        from app.rag.pgvector_retriever import PgVectorRetriever

        return PgVectorRetriever(embedder=FastEmbedTextEmbedder())

    data: dict[str, Any] = load_mock_business_data()
    return InMemoryRetriever(
        documents=data["pricing_documents"],
        embedder=FastEmbedTextEmbedder(),
    )


def retrieve_pricing_documents(
    query: str,
    *,
    limit: int = 3,
    retriever: Retriever | None = None,
) -> list[dict[str, object]]:
    active_retriever = retriever or get_default_retriever()
    matches = []
    seen_source_ids = set()
    for chunk in active_retriever.retrieve(query, limit=limit):
        if chunk.source_id in seen_source_ids:
            continue
        seen_source_ids.add(chunk.source_id)
        matches.append({
            "source_id": chunk.source_id,
            "document_id": chunk.source_id,
            "client": chunk.client,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "score": round(chunk.score, 4),
        })
    return matches
