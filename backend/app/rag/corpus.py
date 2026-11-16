"""Retrieval corpora and how each one is chunked.

``pricing-mock`` is the fictional business data used by the workflow tools.
``stack-docs`` is the documentation corpus built by ``app.rag.build_corpus``
(see ``data/THIRD_PARTY_NOTICES.md`` for sources and licenses).
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from app.workflow_engine.mock_data import load_mock_business_data


PRICING_CORPUS = "pricing-mock"
DOCS_CORPUS = "stack-docs"
DOCS_CORPUS_PATH = Path(__file__).parent / "data" / "stack_docs_corpus.v1.json"

ChunkUnit = Literal["words", "tokens"]


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    load_documents: Callable[[], list[dict[str, str]]]
    # pricing-mock keeps the original word chunker so v1 results stay comparable.
    chunk_unit: ChunkUnit


@lru_cache(maxsize=1)
def load_docs_corpus_file() -> dict[str, Any]:
    return json.loads(DOCS_CORPUS_PATH.read_text(encoding="utf-8"))


def load_docs_documents() -> list[dict[str, str]]:
    """Docs corpus in the shared document shape (client = collection)."""
    return [
        {
            "document_id": document["document_id"],
            "client": document["collection"],
            "title": document["title"],
            "url": document["url"],
            "content": document["content"],
        }
        for document in load_docs_corpus_file()["documents"]
    ]


def load_docs_documents_by_ids(source_ids: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(source_ids)
    return [
        document
        for document in load_docs_documents()
        if document["document_id"] in wanted
    ]


def load_pricing_documents() -> list[dict[str, str]]:
    return load_mock_business_data()["pricing_documents"]


CORPORA: dict[str, CorpusSpec] = {
    PRICING_CORPUS: CorpusSpec(PRICING_CORPUS, load_pricing_documents, "words"),
    DOCS_CORPUS: CorpusSpec(DOCS_CORPUS, load_docs_documents, "tokens"),
}
