"""Check the v2 RAG dataset against the corpus it claims to describe."""

import json
import re
from pathlib import Path
from typing import Any

from app.rag.corpus import load_docs_corpus_file


DATASET_V2_PATH = Path(__file__).with_name("dataset.v2.json")
CATEGORIES = {"easy", "paraphrased", "multi_hop", "unanswerable"}


def load_dataset_v2(path: Path = DATASET_V2_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return " ".join(text.split())


def find_dataset_problems(
    dataset: dict[str, Any],
    documents: dict[str, str] | None = None,
) -> list[str]:
    """Return human-readable problems; an empty list means the dataset is consistent."""
    if documents is None:
        documents = {
            document["document_id"]: document["content"]
            for document in load_docs_corpus_file()["documents"]
        }
    normalized = {doc_id: _normalize(content) for doc_id, content in documents.items()}
    problems: list[str] = []
    seen_ids: set[str] = set()

    for query in dataset["queries"]:
        query_id = query["id"]
        if query_id in seen_ids:
            problems.append(f"{query_id}: duplicate id")
        seen_ids.add(query_id)

        if query["category"] not in CATEGORIES:
            problems.append(f"{query_id}: unknown category {query['category']}")
        if not query["question"].strip() or not query["reference_answer"].strip():
            problems.append(f"{query_id}: empty question or reference answer")

        relevant = query["relevant_source_ids"]
        unknown = [source_id for source_id in relevant if source_id not in documents]
        if unknown:
            problems.append(f"{query_id}: unknown relevant ids {unknown}")

        if query["category"] == "unanswerable":
            if relevant:
                problems.append(f"{query_id}: unanswerable query lists relevant ids")
            pattern = re.compile(query.get("absence_pattern", ""), re.IGNORECASE)
            if not query.get("absence_pattern"):
                problems.append(f"{query_id}: unanswerable query needs absence_pattern")
                continue
            hits = [doc_id for doc_id, content in documents.items() if pattern.search(content)]
            if hits:
                problems.append(f"{query_id}: absence_pattern found in {hits[:5]}")
            continue

        if not relevant:
            problems.append(f"{query_id}: answerable query has no relevant ids")
        if query["category"] == "multi_hop" and len(relevant) < 2:
            problems.append(f"{query_id}: multi-hop query needs two or more relevant ids")

        evidence = query.get("evidence", [])
        if not evidence:
            problems.append(f"{query_id}: no evidence quotes")
        for item in evidence:
            source_id = item["source_id"]
            if source_id not in relevant:
                problems.append(f"{query_id}: evidence source {source_id} not in relevant ids")
            content = normalized.get(source_id)
            if content is None or _normalize(item["quote"]) not in content:
                problems.append(f"{query_id}: quote not found in {source_id}: {item['quote'][:70]!r}")

        if query["category"] == "multi_hop":
            evidenced = {item["source_id"] for item in evidence}
            if len(evidenced) < 2:
                problems.append(f"{query_id}: multi-hop evidence must cite two documents")

    return problems
