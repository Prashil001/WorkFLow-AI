import re

import pytest

from app.rag.build_corpus import clean_markdown
from app.rag.corpus import (
    DOCS_CORPUS_PATH,
    load_docs_corpus_file,
    load_docs_documents,
    load_docs_documents_by_ids,
)
from app.rag.ingest import chunk_corpus_documents, default_chunking
from app.workflow_engine.rag import (
    EMBEDDING_MODEL_NAME,
    FastEmbedTextEmbedder,
    chunk_documents_by_tokens,
)


class WhitespaceCounter:
    """Offline stand-in for the WordPiece tokenizer: one token per word."""

    def count(self, text):
        return len(text.split())


def _doc(content, title="Guide", client="Docs"):
    return {"document_id": "doc-1", "client": client, "title": title, "content": content}


def _body(chunk):
    return chunk.text.split("\n\n", 1)[1]


def _sentences(count, words=6):
    return " ".join(
        f"Sentence {index} " + " ".join(["word"] * (words - 3)) + "."
        for index in range(count)
    )


# --- token / sentence-aware chunking -----------------------------------------


def test_token_chunks_respect_budget_and_end_on_sentence_boundaries():
    chunks = chunk_documents_by_tokens(
        [_doc(_sentences(30))],
        WhitespaceCounter(),
        chunk_tokens=40,
        overlap_tokens=6,
    )

    assert len(chunks) > 3
    for chunk in chunks:
        body = _body(chunk)
        assert WhitespaceCounter().count(body) <= 40
        assert body.startswith("Sentence ")
        assert body.endswith(".")
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_token_chunks_overlap_with_whole_trailing_sentences():
    chunks = chunk_documents_by_tokens(
        [_doc(_sentences(20))],
        WhitespaceCounter(),
        chunk_tokens=30,
        overlap_tokens=6,
    )

    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = _body(previous).split(". ")[-1]
        assert _body(current).startswith(last_sentence.rstrip("."))


def test_zero_overlap_produces_disjoint_chunks():
    content = _sentences(20)
    chunks = chunk_documents_by_tokens(
        [_doc(content)],
        WhitespaceCounter(),
        chunk_tokens=30,
        overlap_tokens=0,
    )

    assert len(chunks) > 1
    assert " ".join(_body(chunk) for chunk in chunks) == content


def test_code_blocks_stay_intact_and_headings_label_chunks():
    content = (
        "Intro sentence here.\n\n"
        "## Install\n\n"
        "Run the command below.\n\n"
        "```bash\npip install fastapi\nuvicorn main:app\n```\n\n"
        "That is all."
    )
    chunks = chunk_documents_by_tokens(
        [_doc(content, title="Setup", client="FastAPI")],
        WhitespaceCounter(),
        chunk_tokens=200,
        overlap_tokens=10,
    )

    assert len(chunks) == 1
    assert chunks[0].text.startswith("FastAPI | Setup\n\n")
    assert "```bash\npip install fastapi\nuvicorn main:app\n```" in chunks[0].text


def test_chunk_header_names_the_section_where_the_chunk_starts():
    content = "## First\n\n" + _sentences(10) + "\n\n## Second\n\n" + _sentences(10)
    chunks = chunk_documents_by_tokens(
        [_doc(content)],
        WhitespaceCounter(),
        chunk_tokens=40,
        overlap_tokens=0,
    )

    headers = [chunk.text.split("\n\n", 1)[0] for chunk in chunks]
    assert headers[0] == "Docs | Guide > First"
    assert headers[-1] == "Docs | Guide > Second"


def test_oversized_units_are_split_to_fit_the_budget():
    long_sentence = " ".join(f"w{index}" for index in range(100)) + "."
    long_code = "```\n" + "\n".join(f"line {index} of code" for index in range(40)) + "\n```"
    chunks = chunk_documents_by_tokens(
        [_doc(long_sentence + "\n\n" + long_code)],
        WhitespaceCounter(),
        chunk_tokens=30,
        overlap_tokens=0,
    )

    assert all(WhitespaceCounter().count(_body(chunk)) <= 30 for chunk in chunks)
    joined = " ".join(_body(chunk) for chunk in chunks)
    assert "w0" in joined and "w99." in joined and "line 39 of code" in joined


@pytest.mark.parametrize(
    ("chunk_tokens", "overlap_tokens"),
    [(8, 0), (100, 100), (100, -1)],
)
def test_token_chunker_rejects_invalid_settings(chunk_tokens, overlap_tokens):
    with pytest.raises(ValueError):
        chunk_documents_by_tokens(
            [_doc("x.")],
            WhitespaceCounter(),
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )


def test_corpus_chunker_dispatch_requires_token_counter_for_tokens():
    with pytest.raises(ValueError, match="token counter"):
        chunk_corpus_documents([_doc("x.")], chunk_unit="tokens", chunk_size=100, overlap=10)

    word_chunks = chunk_corpus_documents(
        [_doc("one two three four five")],
        chunk_unit="words",
        chunk_size=3,
        overlap=1,
    )
    assert [chunk.chunk_index for chunk in word_chunks] == [0, 1]


def test_default_token_chunking_comes_from_settings(monkeypatch):
    import app.core.config as config_module

    class FakeSettings:
        rag_chunk_tokens = 300
        rag_chunk_overlap_tokens = 45

    monkeypatch.setattr(config_module, "get_settings", lambda: FakeSettings())

    assert default_chunking("tokens") == (300, 45)
    assert default_chunking("words") == (22, 5)


def test_model_identity_includes_snapshot_revision():
    class Inner:
        _model_dir = "/cache/models--x/snapshots/aa8f8b060edb00e03bfdd08813a2949946c8ba55"

    class Outer:
        model = Inner()

    embedder = object.__new__(FastEmbedTextEmbedder)
    embedder._model = Outer()

    assert embedder.model_identity == f"{EMBEDDING_MODEL_NAME}@aa8f8b060edb"


# --- markdown cleaning ---------------------------------------------------------


def test_clean_markdown_strips_build_syntax_and_keeps_prose():
    markdown = (
        "---\ntitle: ignored\n---\n"
        "# Path Parameters { #path-parameters }\n\n"
        "Use <dfn title=\"x\">conversion</dfn> with [links](https://x.dev){.internal-link}.\n\n"
        "{* ../../docs_src/app.py hl[3] *}\n\n"
        "{!> ../../docs_src/app.js!}\n\n"
        "/// tip | Pro tip\n\nKeep this text.\n\n///\n\n"
        "/// version-added | v2.10\n\n///\n\n"
        "??? api \"API Documentation\"\n    [`x`][pydantic.x]<br>\n\n"
        "!!! warning \"Careful\"\n    Indented warning.\n\n"
        "[![badge](https://img)](https://ci)\n\n"
        "```python\n# not a title\nx = 1\n```\n"
    )

    title, body = clean_markdown(markdown)

    assert title == "Path Parameters"
    assert "Use conversion with links." in body
    assert "Pro tip:" in body and "Keep this text." in body
    assert "Version added: v2.10" in body
    assert "Careful:" in body and "Indented warning." in body
    assert "# not a title" in body
    for leftover in ("{*", "{!", "///", "???", "!!!", "<dfn", "](", "{ #"):
        assert leftover not in body


def test_clean_markdown_ignores_comment_lines_in_code_when_finding_title():
    title, _ = clean_markdown("Intro.\n\n```python\n# returns a dictionary:\n```\n")

    assert title is None


# --- the committed corpus ----------------------------------------------------


_CITATION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def test_docs_corpus_size_provenance_and_ids():
    corpus = load_docs_corpus_file()
    documents = corpus["documents"]

    assert 200 <= len(documents) <= 500
    assert corpus["document_count"] == len(documents)
    ids = [document["document_id"] for document in documents]
    assert len(ids) == len(set(ids))
    assert all(_CITATION_ID.match(doc_id) and len(doc_id) <= 128 for doc_id in ids)

    notices = (DOCS_CORPUS_PATH.parent / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert {source["collection"] for source in corpus["sources"]} == {
        "FastAPI", "Pydantic", "SQLModel",
    }
    for source in corpus["sources"]:
        assert source["license"] == "MIT"
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["commit"] in notices
    assert notices.count("Permission is hereby granted") == len(corpus["sources"])


def test_docs_corpus_has_no_leftover_build_markup():
    leftovers = re.compile(r"\{\*|\{!|^/{3}|^\s*!!!|^\s*\?\?\?|\[\]\(|<svg", re.MULTILINE)
    offenders = [
        document["document_id"]
        for document in load_docs_corpus_file()["documents"]
        if leftovers.search(document["content"])
    ]

    assert offenders == []


def test_docs_documents_use_shared_shape_and_lookup_by_id():
    documents = load_docs_documents()
    first = documents[0]

    assert set(first) == {"document_id", "client", "title", "url", "content"}
    assert first["url"].startswith("https://")
    assert load_docs_documents_by_ids([first["document_id"], "missing"]) == [first]
