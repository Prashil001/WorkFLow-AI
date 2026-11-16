import json

import pytest
from sqlalchemy import exc, func, select, text

from app.models.document_chunk import EMBEDDING_DIMENSIONS, DocumentChunkRecord
from app.rag.ingest import (
    _embed_chunks,
    chunk_content_hash,
    ingest_documents,
    plan_ingestion,
)
from app.rag.pgvector_retriever import PgVectorRetriever
from app.workflow_engine import rag as rag_module
from app.workflow_engine.mock_data import load_mock_business_data
from app.workflow_engine.mock_tools import search_pricing_documents
from app.workflow_engine.rag import (
    DocumentChunk,
    InMemoryRetriever,
    RetrievedChunk,
    rank_retrieved_chunks,
)


TEST_MODEL = "test-keyword-embedder"


class PaddedKeywordEmbedder:
    """Deterministic 384-dim embedder so vectors fit the vector(384) column."""

    def _vector(self, text):
        normalized = text.lower()
        head = [0.0, 0.0, 0.0]
        if any(word in normalized for word in ("northstar", "standard pricing", "implementation packages")):
            head = [1.0, 0.0, 0.0]
        elif any(word in normalized for word in ("cedar", "retail", "training")):
            head = [0.0, 1.0, 0.0]
        elif any(word in normalized for word in ("bluebird", "healthcare", "onboarding")):
            head = [0.0, 0.0, 1.0]
        # A small constant tail keeps every vector non-zero for cosine distance.
        return head + [0.001] * (EMBEDDING_DIMENSIONS - len(head))

    def embed(self, documents):
        return [self._vector(document) for document in documents]

    def query_embed(self, query):
        return [self._vector(query)]


def _chunk(source_id="doc-1", chunk_index=0, text="alpha beta"):
    return DocumentChunk(
        source_id=source_id,
        client="Example Co",
        title="Guide",
        chunk_index=chunk_index,
        text=text,
    )


def _corpus():
    return [dict(document) for document in load_mock_business_data()["pricing_documents"]]


# --- Unit tests: no database -------------------------------------------------


def test_plan_ingestion_classifies_new_changed_unchanged_and_stale_chunks():
    unchanged = _chunk("doc-1", 0, "same text")
    changed = _chunk("doc-1", 1, "new text")
    new = _chunk("doc-2", 0, "brand new")
    existing = {
        ("doc-1", 0): chunk_content_hash(unchanged, TEST_MODEL),
        ("doc-1", 1): chunk_content_hash(_chunk("doc-1", 1, "old text"), TEST_MODEL),
        ("doc-9", 0): "stale-hash",
    }

    plan = plan_ingestion([unchanged, changed, new], existing, TEST_MODEL)

    assert plan.to_insert == [new]
    assert plan.to_update == [changed]
    assert plan.unchanged_count == 1
    assert plan.stale_keys == [("doc-9", 0)]


def test_changing_embedding_model_forces_reembedding():
    chunk = _chunk()
    existing = {("doc-1", 0): chunk_content_hash(chunk, "old-model")}

    plan = plan_ingestion([chunk], existing, "new-model")

    assert plan.to_update == [chunk]
    assert plan.unchanged_count == 0


def test_plan_ingestion_rejects_duplicate_chunk_keys():
    with pytest.raises(ValueError, match="Duplicate chunk key"):
        plan_ingestion([_chunk(), _chunk()], {}, TEST_MODEL)


def test_embed_chunks_rejects_wrong_dimensions():
    class TinyEmbedder:
        def embed(self, documents):
            return [[1.0, 0.0] for _ in documents]

    with pytest.raises(ValueError, match="384-dimensional"):
        _embed_chunks([_chunk()], TinyEmbedder())


def test_rank_retrieved_chunks_applies_threshold_tiebreak_and_limit():
    def retrieved(source_id, chunk_index, score):
        return RetrievedChunk(source_id, "c", "t", chunk_index, "x", score)

    ranked = rank_retrieved_chunks(
        [
            retrieved("b", 0, 0.9),
            retrieved("a", 1, 0.9),
            retrieved("a", 0, 0.9),
            retrieved("c", 0, 0.5),
        ],
        minimum_score=0.6,
        limit=2,
    )

    assert [(c.source_id, c.chunk_index) for c in ranked] == [("a", 0), ("a", 1)]


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    [("memory", InMemoryRetriever), ("pgvector", PgVectorRetriever)],
)
def test_default_retriever_is_selected_by_config(monkeypatch, backend, expected_type):
    import app.core.config as config_module

    class FakeSettings:
        rag_retriever = backend

    monkeypatch.setattr(config_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(rag_module, "FastEmbedTextEmbedder", PaddedKeywordEmbedder)
    rag_module.get_default_retriever.cache_clear()
    try:
        assert isinstance(rag_module.get_default_retriever(), expected_type)
    finally:
        rag_module.get_default_retriever.cache_clear()


# --- Integration tests: PostgreSQL + pgvector, always rolled back ------------


@pytest.fixture
def pg_connection():
    """Yield a connection inside a transaction that is always rolled back."""
    from app.db.database import engine

    try:
        connection = engine.connect()
    except exc.OperationalError as error:
        pytest.skip(f"PostgreSQL is not reachable: {type(error).__name__}")

    transaction = connection.begin()
    try:
        has_table = connection.execute(
            text("SELECT to_regclass('public.document_chunks') IS NOT NULL")
        ).scalar_one()
        if not has_table:
            pytest.skip("document_chunks table missing; run alembic upgrade head")
        # Start from an empty table inside this transaction only.
        connection.execute(text("DELETE FROM document_chunks"))
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _ingest(connection, documents, **kwargs):
    return ingest_documents(
        documents,
        connection=connection,
        embedder=PaddedKeywordEmbedder(),
        embedding_model=TEST_MODEL,
        **kwargs,
    )


def _stored_rows(connection):
    return connection.execute(
        select(func.count()).select_from(DocumentChunkRecord)
    ).scalar_one()


def test_ingestion_is_idempotent(pg_connection):
    first = _ingest(pg_connection, _corpus())
    second = _ingest(pg_connection, _corpus())

    assert first.inserted == first.chunk_count > 0
    assert (second.inserted, second.updated, second.deleted) == (0, 0, 0)
    assert second.unchanged == first.chunk_count
    assert _stored_rows(pg_connection) == first.chunk_count


def test_ingestion_updates_changed_chunks_and_deletes_stale_ones(pg_connection):
    corpus = _corpus()
    first = _ingest(pg_connection, corpus)

    corpus[0]["content"] = "Northstar Labs revised standard pricing."
    second = _ingest(pg_connection, corpus[:2])

    removed_chunks = first.chunk_count - second.chunk_count
    assert second.updated == 1
    assert second.deleted == removed_chunks
    assert _stored_rows(pg_connection) == second.chunk_count
    remaining_sources = set(pg_connection.execute(
        select(DocumentChunkRecord.source_id).distinct()
    ).scalars())
    assert remaining_sources == {corpus[0]["document_id"], corpus[1]["document_id"]}


def test_dry_run_reports_without_writing(pg_connection):
    report = _ingest(pg_connection, _corpus(), dry_run=True)

    assert report.inserted > 0
    assert _stored_rows(pg_connection) == 0


def test_ingested_rows_have_full_text_search_vectors(pg_connection):
    _ingest(pg_connection, _corpus())

    matched_sources = set(pg_connection.execute(
        select(DocumentChunkRecord.source_id).where(
            DocumentChunkRecord.search_vector.op("@@")(
                func.plainto_tsquery("english", "Northstar implementation")
            )
        )
    ).scalars())

    assert matched_sources == {"mock-pricing-001"}


def test_pgvector_retriever_matches_in_memory_retriever(pg_connection):
    _ingest(pg_connection, _corpus())
    embedder = PaddedKeywordEmbedder()
    pg_retriever = PgVectorRetriever(embedder, bind=pg_connection, minimum_score=0.2)
    memory_retriever = InMemoryRetriever(_corpus(), embedder, minimum_score=0.2)

    for query in (
        "Northstar Labs standard pricing",
        "Cedar retail training",
        "Bluebird healthcare onboarding",
    ):
        pg_results = pg_retriever.retrieve(query, limit=5)
        memory_results = memory_retriever.retrieve(query, limit=5)
        assert [(r.source_id, r.chunk_index) for r in pg_results] == [
            (r.source_id, r.chunk_index) for r in memory_results
        ]
        assert [r.score for r in pg_results] == pytest.approx(
            [r.score for r in memory_results], abs=1e-5
        )

    assert pg_retriever.retrieve("Northstar Labs standard pricing")[0].source_id == (
        "mock-pricing-001"
    )
    assert pg_retriever.retrieve("quantum yacht") == []
    assert pg_retriever.retrieve("   ") == []


def _docs_subset(count=40):
    from app.rag.corpus import load_docs_documents

    return load_docs_documents()[:count]


def test_corpora_are_isolated_for_sync_and_retrieval(pg_connection):
    from app.rag.corpus import DOCS_CORPUS, PRICING_CORPUS

    pricing = _ingest(pg_connection, _corpus(), corpus=PRICING_CORPUS)
    docs = _ingest(
        pg_connection,
        _docs_subset(),
        corpus=DOCS_CORPUS,
        chunk_unit="tokens",
        chunk_size=120,
        overlap=18,
        token_counter=WhitespaceCounter(),
    )
    # Re-syncing docs with fewer documents must not touch pricing rows.
    resync = _ingest(
        pg_connection,
        _docs_subset(10),
        corpus=DOCS_CORPUS,
        chunk_unit="tokens",
        chunk_size=120,
        overlap=18,
        token_counter=WhitespaceCounter(),
    )

    counts = dict(pg_connection.execute(
        select(DocumentChunkRecord.corpus, func.count()).group_by(DocumentChunkRecord.corpus)
    ).all())
    assert counts[PRICING_CORPUS] == pricing.chunk_count
    assert counts[DOCS_CORPUS] == resync.chunk_count < docs.chunk_count
    assert resync.deleted == docs.chunk_count - resync.chunk_count

    urls = set(pg_connection.execute(
        select(DocumentChunkRecord.source_url).where(DocumentChunkRecord.corpus == DOCS_CORPUS)
    ).scalars())
    assert all(url and url.startswith("https://") for url in urls)

    # Pricing rows are a small minority; the iterative HNSW scan must still
    # return them in order when filtering by corpus.
    pricing_retriever = PgVectorRetriever(
        PaddedKeywordEmbedder(),
        corpus=PRICING_CORPUS,
        bind=pg_connection,
        minimum_score=0.2,
    )
    matches = pricing_retriever.retrieve("Northstar Labs standard pricing", limit=5)
    assert matches and all(m.source_id.startswith("mock-pricing-") for m in matches)
    assert matches[0].source_id == "mock-pricing-001"

    docs_retriever = PgVectorRetriever(
        PaddedKeywordEmbedder(),
        corpus=DOCS_CORPUS,
        bind=pg_connection,
        minimum_score=0.0,
    )
    doc_ids = {doc["document_id"] for doc in _docs_subset(10)}
    assert all(m.source_id in doc_ids for m in docs_retriever.retrieve("anything", limit=10))


class WhitespaceCounter:
    def count(self, text):
        return len(text.split())


def test_full_text_retriever_ranks_matches_within_its_corpus(pg_connection):
    from app.rag.corpus import PRICING_CORPUS
    from app.rag.hybrid import HybridRetriever, PgFullTextRetriever

    _ingest(pg_connection, _corpus(), corpus=PRICING_CORPUS)
    lexical = PgFullTextRetriever(corpus=PRICING_CORPUS, bind=pg_connection)

    matches = lexical.retrieve("Northstar Labs implementation packages", limit=5)

    assert matches[0].source_id == "mock-pricing-001"
    assert all(m.source_id.startswith("mock-pricing-") for m in matches)
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)
    # Any term matching is enough (OR semantics).
    assert lexical.retrieve("Cedar spaceship")[0].source_id == "mock-pricing-002"
    # Stop words only, empty input, and tsquery syntax characters are safe.
    assert lexical.retrieve("the and of") == []
    assert lexical.retrieve("  ") == []
    assert lexical.retrieve("Northstar & | ! :* ( ) '")[0].source_id == "mock-pricing-001"
    assert PgFullTextRetriever(corpus="other", bind=pg_connection).retrieve("Northstar") == []

    hybrid = HybridRetriever(
        PgVectorRetriever(PaddedKeywordEmbedder(), corpus=PRICING_CORPUS,
                          bind=pg_connection, minimum_score=0.0),
        lexical,
    )
    assert hybrid.retrieve("Northstar Labs standard pricing")[0].source_id == "mock-pricing-001"


def test_pgvector_retriever_plugs_into_pricing_search_tool(pg_connection):
    _ingest(pg_connection, _corpus())
    retriever = PgVectorRetriever(
        PaddedKeywordEmbedder(),
        bind=pg_connection,
        minimum_score=0.2,
    )

    payload = json.loads(search_pricing_documents(
        "Northstar Labs standard pricing",
        retriever=retriever,
    ))

    assert [match["source_id"] for match in payload["matches"]] == ["mock-pricing-001"]
