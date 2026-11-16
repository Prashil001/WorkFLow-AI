"""Chunk, embed, and sync a retrieval corpus into PostgreSQL (pgvector).

The run is idempotent and scoped to one corpus: chunks whose content and
embedding model are unchanged are skipped without re-embedding, changed
chunks are updated in place, and chunks that no longer exist in that corpus
are deleted. The sync runs in one transaction under a per-corpus advisory
lock, so concurrent runs for the same corpus serialize.

    uv run python -m app.rag.ingest --corpus stack-docs
    uv run python -m app.rag.ingest --corpus stack-docs --chunk-size 300 --overlap 45
"""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, func, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.models.document_chunk import EMBEDDING_DIMENSIONS, DocumentChunkRecord
from app.rag.corpus import CORPORA, PRICING_CORPUS, ChunkUnit
from app.workflow_engine.rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    DocumentChunk,
    TextEmbedder,
    TokenCounter,
    chunk_documents,
    chunk_documents_by_tokens,
)


# Arbitrary constant; combined with a hash of the corpus name per lock.
_INGEST_ADVISORY_LOCK_KEY = 7_340_001

ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class IngestionPlan:
    to_insert: list[DocumentChunk]
    to_update: list[DocumentChunk]
    unchanged_count: int
    stale_keys: list[ChunkKey]


@dataclass(frozen=True)
class IngestionReport:
    corpus: str
    embedding_model: str
    chunk_unit: ChunkUnit
    chunk_size: int
    overlap: int
    document_count: int
    chunk_count: int
    inserted: int
    updated: int
    unchanged: int
    deleted: int
    dry_run: bool


def chunk_content_hash(
    chunk: DocumentChunk,
    embedding_model: str,
    source_url: str | None = None,
) -> str:
    payload = "\x1f".join((
        embedding_model,
        chunk.source_id,
        chunk.client,
        chunk.title,
        source_url or "",
        chunk.text,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_ingestion(
    chunks: Sequence[DocumentChunk],
    existing_hashes: Mapping[ChunkKey, str],
    embedding_model: str,
    source_urls: Mapping[str, str | None] | None = None,
) -> IngestionPlan:
    """Diff the desired chunks against what is stored, without touching the DB."""
    source_urls = source_urls or {}
    to_insert: list[DocumentChunk] = []
    to_update: list[DocumentChunk] = []
    unchanged_count = 0
    desired_keys: set[ChunkKey] = set()

    for chunk in chunks:
        key = (chunk.source_id, chunk.chunk_index)
        if key in desired_keys:
            raise ValueError(f"Duplicate chunk key in corpus: {key}")
        desired_keys.add(key)

        stored_hash = existing_hashes.get(key)
        desired_hash = chunk_content_hash(
            chunk,
            embedding_model,
            source_urls.get(chunk.source_id),
        )
        if stored_hash is None:
            to_insert.append(chunk)
        elif stored_hash != desired_hash:
            to_update.append(chunk)
        else:
            unchanged_count += 1

    stale_keys = sorted(set(existing_hashes) - desired_keys)
    return IngestionPlan(
        to_insert=to_insert,
        to_update=to_update,
        unchanged_count=unchanged_count,
        stale_keys=stale_keys,
    )


def chunk_corpus_documents(
    documents: list[dict[str, str]],
    *,
    chunk_unit: ChunkUnit,
    chunk_size: int,
    overlap: int,
    token_counter: TokenCounter | None = None,
) -> list[DocumentChunk]:
    if chunk_unit == "words":
        return chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if token_counter is None:
        raise ValueError("Token chunking requires a token counter")
    return chunk_documents_by_tokens(
        documents,
        token_counter,
        chunk_tokens=chunk_size,
        overlap_tokens=overlap,
    )


def ingest_documents(
    documents: list[dict[str, str]],
    *,
    connection: Connection,
    embedder: TextEmbedder,
    corpus: str = PRICING_CORPUS,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    chunk_unit: ChunkUnit = "words",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    token_counter: TokenCounter | None = None,
    dry_run: bool = False,
) -> IngestionReport:
    """Sync one corpus in document_chunks to exactly match its chunked documents.

    The caller owns the transaction on ``connection``.
    """
    connection.execute(
        text("SELECT pg_advisory_xact_lock(:key, hashtext(:corpus))"),
        {"key": _INGEST_ADVISORY_LOCK_KEY, "corpus": corpus},
    )

    chunks = chunk_corpus_documents(
        documents,
        chunk_unit=chunk_unit,
        chunk_size=chunk_size,
        overlap=overlap,
        token_counter=token_counter,
    )
    source_urls = {
        document["document_id"]: document.get("url")
        for document in documents
    }
    existing_hashes: dict[ChunkKey, str] = {
        (row.source_id, row.chunk_index): row.content_hash
        for row in connection.execute(
            select(
                DocumentChunkRecord.source_id,
                DocumentChunkRecord.chunk_index,
                DocumentChunkRecord.content_hash,
            ).where(DocumentChunkRecord.corpus == corpus)
        )
    }
    plan = plan_ingestion(chunks, existing_hashes, embedding_model, source_urls)

    if not dry_run:
        changed = plan.to_insert + plan.to_update
        if changed:
            vectors = _embed_chunks(changed, embedder)
            _upsert_chunks(
                connection,
                corpus,
                changed,
                vectors,
                embedding_model,
                source_urls,
            )
        if plan.stale_keys:
            connection.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.corpus == corpus,
                    tuple_(
                        DocumentChunkRecord.source_id,
                        DocumentChunkRecord.chunk_index,
                    ).in_(plan.stale_keys),
                )
            )

    return IngestionReport(
        corpus=corpus,
        embedding_model=embedding_model,
        chunk_unit=chunk_unit,
        chunk_size=chunk_size,
        overlap=overlap,
        document_count=len(documents),
        chunk_count=len(chunks),
        inserted=len(plan.to_insert),
        updated=len(plan.to_update),
        unchanged=plan.unchanged_count,
        deleted=len(plan.stale_keys),
        dry_run=dry_run,
    )


def _embed_chunks(
    chunks: Sequence[DocumentChunk],
    embedder: TextEmbedder,
) -> list[list[float]]:
    vectors = [
        [float(value) for value in vector]
        for vector in embedder.embed([chunk.text for chunk in chunks])
    ]
    if len(vectors) != len(chunks):
        raise ValueError("Embedder returned an unexpected number of vectors")
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings, "
                f"got {len(vector)}"
            )
    return vectors


_UPSERT_BATCH_SIZE = 500


def _upsert_chunks(
    connection: Connection,
    corpus: str,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[list[float]],
    embedding_model: str,
    source_urls: Mapping[str, str | None],
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "corpus": corpus,
            "source_id": chunk.source_id,
            "source_url": source_urls.get(chunk.source_id),
            "client": chunk.client,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "content_hash": chunk_content_hash(
                chunk,
                embedding_model,
                source_urls.get(chunk.source_id),
            ),
            "embedding_model": embedding_model,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    statement = insert(DocumentChunkRecord)
    statement = statement.on_conflict_do_update(
        constraint="uq_document_chunks_corpus_source_chunk",
        set_={
            "source_url": statement.excluded.source_url,
            "client": statement.excluded.client,
            "title": statement.excluded.title,
            "text": statement.excluded.text,
            "content_hash": statement.excluded.content_hash,
            "embedding_model": statement.excluded.embedding_model,
            "embedding": statement.excluded.embedding,
            "updated_at": func.now(),
        },
    )
    for start in range(0, len(rows), _UPSERT_BATCH_SIZE):
        connection.execute(statement, rows[start : start + _UPSERT_BATCH_SIZE])


def default_chunking(chunk_unit: ChunkUnit) -> tuple[int, int]:
    if chunk_unit == "words":
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    from app.core.config import get_settings

    settings = get_settings()
    return settings.rag_chunk_tokens, settings.rag_chunk_overlap_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", choices=sorted(CORPORA), default=PRICING_CORPUS)
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk budget in the corpus unit (words for pricing-mock, "
        "tokens for stack-docs; tokens default to RAG_CHUNK_TOKENS).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Overlap budget in the same unit (default RAG_CHUNK_OVERLAP_TOKENS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without embedding or writing.",
    )
    args = parser.parse_args()

    from app.db.database import engine
    from app.workflow_engine.rag import FastEmbedTextEmbedder

    spec = CORPORA[args.corpus]
    default_size, default_overlap = default_chunking(spec.chunk_unit)
    embedder = FastEmbedTextEmbedder()
    with engine.begin() as connection:
        report = ingest_documents(
            spec.load_documents(),
            connection=connection,
            embedder=embedder,
            embedding_model=embedder.model_identity,
            corpus=spec.name,
            chunk_unit=spec.chunk_unit,
            chunk_size=args.chunk_size or default_size,
            overlap=default_overlap if args.overlap is None else args.overlap,
            token_counter=(
                embedder.token_counter() if spec.chunk_unit == "tokens" else None
            ),
            dry_run=args.dry_run,
        )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
