import re
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
# bge-small-en-v1.5 truncates at 512 tokens; 400 leaves room for the chunk
# header and special tokens. 60 tokens is 15% overlap.
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 16


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


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class _TextUnit:
    text: str
    tokens: int
    starts_block: bool
    heading: str | None


_HEADING_LINE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
# Split after sentence punctuation when the next sentence looks like a start.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[*])")


def chunk_documents_by_tokens(
    documents: list[dict[str, str]],
    token_counter: TokenCounter,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """Sentence-aware chunking with a token budget and sentence-level overlap.

    Headings, prose sentences, and fenced code blocks are packed in order
    until the next unit would exceed ``chunk_tokens``. The next chunk starts
    with the trailing whole units of the previous one that fit within
    ``overlap_tokens``. Units larger than the budget are split by line, then
    by word. Each chunk is prefixed with "collection | title > section".
    """
    if chunk_tokens < MIN_CHUNK_TOKENS:
        raise ValueError(f"Chunk size must be at least {MIN_CHUNK_TOKENS} tokens")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("Overlap must be non-negative and smaller than the chunk size")

    chunks: list[DocumentChunk] = []
    for document in documents:
        units = _document_units(document["content"], token_counter, chunk_tokens)
        for chunk_index, packed in enumerate(
            _pack_units(units, chunk_tokens, overlap_tokens)
        ):
            chunks.append(DocumentChunk(
                source_id=document["document_id"],
                client=document["client"],
                title=document["title"],
                chunk_index=chunk_index,
                text=_render_chunk(document, packed),
            ))
    return chunks


def _document_units(
    content: str,
    token_counter: TokenCounter,
    chunk_tokens: int,
) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    heading: str | None = None

    def add(text: str, starts_block: bool) -> None:
        text = text.strip()
        if not text:
            return
        tokens = token_counter.count(text)
        if tokens <= chunk_tokens:
            units.append(_TextUnit(text, tokens, starts_block, heading))
            return
        for index, piece in enumerate(
            _split_oversized(text, token_counter, chunk_tokens)
        ):
            units.append(_TextUnit(
                piece,
                token_counter.count(piece),
                starts_block and index == 0,
                heading,
            ))

    paragraph: list[str] = []
    code: list[str] | None = None

    def flush_paragraph() -> None:
        if not paragraph:
            return
        sentences = _SENTENCE_BOUNDARY.split(" ".join(paragraph))
        for index, sentence in enumerate(sentences):
            add(sentence, index == 0)
        paragraph.clear()

    for line in content.splitlines():
        if code is not None:
            code.append(line)
            if _FENCE_LINE.match(line):
                add("\n".join(code), True)
                code = None
            continue
        if _FENCE_LINE.match(line):
            flush_paragraph()
            code = [line]
            continue
        heading_match = _HEADING_LINE.match(line)
        if heading_match:
            flush_paragraph()
            heading = heading_match.group(1).strip()
            add(heading, True)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    if code is not None:
        add("\n".join(code), True)
    return units


def _split_oversized(
    text: str,
    token_counter: TokenCounter,
    chunk_tokens: int,
) -> list[str]:
    """Split one unit that exceeds the budget: by line first, then by word."""
    separator, parts = ("\n", text.split("\n")) if "\n" in text else (" ", text.split())
    pieces: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = separator.join([*current, part])
        if current and token_counter.count(candidate) > chunk_tokens:
            pieces.append(separator.join(current))
            current = []
        if not current and token_counter.count(part) > chunk_tokens:
            if separator == "\n":
                pieces.extend(_split_oversized(part, token_counter, chunk_tokens))
            else:
                pieces.append(part)  # a single giant token run; keep as-is
            continue
        current.append(part)
    if current:
        pieces.append(separator.join(current))
    return [piece for piece in pieces if piece.strip()]


def _pack_units(
    units: list[_TextUnit],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[list[_TextUnit]]:
    packed: list[list[_TextUnit]] = []
    current: list[_TextUnit] = []
    current_tokens = 0
    for unit in units:
        if current and current_tokens + unit.tokens > chunk_tokens:
            packed.append(current)
            tail: list[_TextUnit] = []
            tail_tokens = 0
            for previous in reversed(current):
                if tail_tokens + previous.tokens > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_tokens += previous.tokens
            while tail and tail_tokens + unit.tokens > chunk_tokens:
                tail_tokens -= tail.pop(0).tokens
            current, current_tokens = tail, tail_tokens
        current.append(unit)
        current_tokens += unit.tokens
    if current:
        packed.append(current)
    return packed


def _render_chunk(document: dict[str, str], units: list[_TextUnit]) -> str:
    header = f"{document['client']} | {document['title']}"
    section = units[0].heading
    if section and section != document["title"]:
        header = f"{header} > {section}"

    body = ""
    for unit in units:
        if body:
            body += "\n\n" if unit.starts_block else " "
        body += unit.text
    return f"{header}\n\n{body}"


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

    def token_counter(self) -> "FastEmbedTokenCounter":
        return FastEmbedTokenCounter(self._model.model.tokenizer)

    @property
    def model_identity(self) -> str:
        """Model name plus the downloaded snapshot revision, when known.

        FastEmbed fetches the latest upstream ONNX snapshot and does not pin a
        revision, so the same model name can produce different vectors after
        a cache refresh. Storing this identity makes ingestion re-embed then.
        """
        from pathlib import Path

        model_dir = getattr(self._model.model, "_model_dir", None)
        if model_dir is None:
            return EMBEDDING_MODEL_NAME
        return f"{EMBEDDING_MODEL_NAME}@{Path(model_dir).name[:12]}"


class FastEmbedTokenCounter:
    """Counts tokens with the embedding model's own WordPiece tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        from tokenizers import Tokenizer

        # Copy so FastEmbed's 512-token truncation and padding do not cap counts.
        self._tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


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
