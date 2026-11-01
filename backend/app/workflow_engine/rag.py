from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any, Protocol

from app.workflow_engine.mock_data import load_mock_business_data


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


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
    chunk_size: int = 22,
    overlap: int = 5,
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
        chunk_size: int = 22,
        overlap: int = 5,
        minimum_score: float = 0.25,
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

        scored_chunks = []
        for chunk, vector in zip(self.chunks, self._vectors, strict=True):
            score = _cosine_similarity(query_vector, vector)
            if score >= self.minimum_score:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                item[1].source_id,
                item[1].chunk_index,
            )
        )

        return [
            RetrievedChunk(
                source_id=chunk.source_id,
                client=chunk.client,
                title=chunk.title,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored_chunks[: max(1, min(limit, 10))]
        ]


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
def get_default_retriever() -> InMemoryRetriever:
    data: dict[str, Any] = load_mock_business_data()
    return InMemoryRetriever(
        documents=data["pricing_documents"],
        embedder=FastEmbedTextEmbedder(),
    )


def retrieve_pricing_documents(
    query: str,
    *,
    limit: int = 3,
    retriever: InMemoryRetriever | None = None,
) -> list[dict[str, object]]:
    active_retriever = retriever or get_default_retriever()
    return [
        {
            "source_id": chunk.source_id,
            "document_id": chunk.source_id,
            "client": chunk.client,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "score": round(chunk.score, 4),
        }
        for chunk in active_retriever.retrieve(query, limit=limit)
    ]
