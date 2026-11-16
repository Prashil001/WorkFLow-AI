"""Chunk, embed, and sync the retrieval corpus into PostgreSQL (pgvector).

The run is idempotent: chunks whose content and embedding model are unchanged
are skipped without re-embedding, changed chunks are updated in place, and
chunks that no longer exist in the corpus are deleted. The whole sync runs in
one transaction guarded by an advisory lock, so concurrent runs serialize.
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
from app.workflow_engine.mock_data import load_mock_business_data
from app.workflow_engine.rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    DocumentChunk,
    TextEmbedder,
    chunk_documents,
)


# Arbitrary constant key so only one ingestion writes document_chunks at a time.
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
    embedding_model: str
    chunk_size: int
    overlap: int
    document_count: int
    chunk_count: int
    inserted: int
    updated: int
    unchanged: int
    deleted: int
    dry_run: bool


def load_corpus_documents() -> list[dict[str, str]]:
    """Return the documents that make up the retrieval corpus."""
    return load_mock_business_data()["pricing_documents"]


def chunk_content_hash(chunk: DocumentChunk, embedding_model: str) -> str:
    payload = "\x1f".join((
        embedding_model,
        chunk.source_id,
        chunk.client,
        chunk.title,
        chunk.text,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_ingestion(
    chunks: Sequence[DocumentChunk],
    existing_hashes: Mapping[ChunkKey, str],
    embedding_model: str,
) -> IngestionPlan:
    """Diff the desired chunks against what is stored, without touching the DB."""
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
        if stored_hash is None:
            to_insert.append(chunk)
        elif stored_hash != chunk_content_hash(chunk, embedding_model):
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


def ingest_documents(
    documents: list[dict[str, str]],
    *,
    connection: Connection,
    embedder: TextEmbedder,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    dry_run: bool = False,
) -> IngestionReport:
    """Sync document_chunks to exactly match the chunked corpus.

    The caller owns the transaction on ``connection``.
    """
    connection.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": _INGEST_ADVISORY_LOCK_KEY},
    )

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    existing_hashes: dict[ChunkKey, str] = {
        (row.source_id, row.chunk_index): row.content_hash
        for row in connection.execute(select(
            DocumentChunkRecord.source_id,
            DocumentChunkRecord.chunk_index,
            DocumentChunkRecord.content_hash,
        ))
    }
    plan = plan_ingestion(chunks, existing_hashes, embedding_model)

    if not dry_run:
        changed = plan.to_insert + plan.to_update
        if changed:
            vectors = _embed_chunks(changed, embedder)
            _upsert_chunks(connection, changed, vectors, embedding_model)
        if plan.stale_keys:
            connection.execute(
                delete(DocumentChunkRecord).where(
                    tuple_(
                        DocumentChunkRecord.source_id,
                        DocumentChunkRecord.chunk_index,
                    ).in_(plan.stale_keys)
                )
            )

    return IngestionReport(
        embedding_model=embedding_model,
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


def _upsert_chunks(
    connection: Connection,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[list[float]],
    embedding_model: str,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "source_id": chunk.source_id,
            "client": chunk.client,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "content_hash": chunk_content_hash(chunk, embedding_model),
            "embedding_model": embedding_model,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    statement = insert(DocumentChunkRecord)
    statement = statement.on_conflict_do_update(
        constraint="uq_document_chunks_source_chunk",
        set_={
            "client": statement.excluded.client,
            "title": statement.excluded.title,
            "text": statement.excluded.text,
            "content_hash": statement.excluded.content_hash,
            "embedding_model": statement.excluded.embedding_model,
            "embedding": statement.excluded.embedding,
            "updated_at": func.now(),
        },
    )
    connection.execute(statement, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without embedding or writing.",
    )
    args = parser.parse_args()

    from app.db.database import engine
    from app.workflow_engine.rag import FastEmbedTextEmbedder

    with engine.begin() as connection:
        report = ingest_documents(
            load_corpus_documents(),
            connection=connection,
            embedder=FastEmbedTextEmbedder(),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            dry_run=args.dry_run,
        )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
