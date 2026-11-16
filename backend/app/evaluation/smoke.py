"""Small end-to-end RAG smoke test for CI.

    uv run python -m app.evaluation.smoke

Ingests a fixed slice of the docs corpus into an isolated ``stack-docs-smoke``
corpus with the real embedding model, runs the vector and hybrid retrievers on
a fixed subset of v2 queries, and exits non-zero if quality falls below floors.

The floors are regression guards set well below what this slice scored when
they were chosen; they are not reported results. The reported metrics come from
``app.evaluation.run_v2`` on the full corpus.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.evaluation.dataset_checks import find_dataset_problems, load_dataset_v2
from app.evaluation.run_v2 import (
    build_hybrid_config,
    build_vector_config,
    evaluate_retrieval,
)
from app.rag.corpus import load_docs_documents


SMOKE_CORPUS = "stack-docs-smoke"
QUERIES_PER_CATEGORY = {"easy": 5, "paraphrased": 5, "multi_hop": 3, "unanswerable": 3}
DISTRACTOR_STRIDE = 4  # every 4th corpus page joins the slice as a distractor
FLOORS = {
    "vector": {"recall@5": 0.70, "mrr@10": 0.60},
    "hybrid": {"recall@5": 0.70, "mrr@10": 0.55},
}


def select_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for category, count in QUERIES_PER_CATEGORY.items():
        selected.extend([q for q in queries if q["category"] == category][:count])
    return selected


def select_documents(
    documents: list[dict[str, str]],
    queries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Every relevant page for the selected queries plus a fixed distractor stride."""
    relevant = {source_id for q in queries for source_id in q["relevant_source_ids"]}
    return [
        document
        for index, document in enumerate(documents)
        if document["document_id"] in relevant or index % DISTRACTOR_STRIDE == 0
    ]


def check_floors(metrics_by_config: dict[str, dict[str, float]]) -> list[str]:
    failures = []
    for config, floors in FLOORS.items():
        for metric, floor in floors.items():
            value = metrics_by_config[config][metric]
            if value < floor:
                failures.append(f"{config} {metric}={value} is below the floor {floor}")
    return failures


def run_smoke() -> dict[str, Any]:
    from app.db.database import engine
    from app.rag.ingest import default_chunking, ingest_documents
    from app.workflow_engine.rag import FastEmbedTextEmbedder

    dataset = load_dataset_v2()
    problems = find_dataset_problems(dataset)
    if problems:
        raise ValueError(f"Dataset failed verification: {problems[:3]}")

    queries = select_queries(dataset["queries"])
    documents = select_documents(load_docs_documents(), queries)
    embedder = FastEmbedTextEmbedder()
    chunk_size, overlap = default_chunking("tokens")
    with engine.begin() as connection:
        ingestion = ingest_documents(
            documents,
            connection=connection,
            embedder=embedder,
            embedding_model=embedder.model_identity,
            corpus=SMOKE_CORPUS,
            chunk_unit="tokens",
            chunk_size=chunk_size,
            overlap=overlap,
            token_counter=embedder.token_counter(),
        )

    results = {}
    for name, builder in (("vector", build_vector_config), ("hybrid", build_hybrid_config)):
        config = builder(SMOKE_CORPUS, embedder)
        report, _ = evaluate_retrieval(queries, config.retriever)
        results[name] = {
            "metrics": report["metrics"],
            "latency": {k: v for k, v in report["latency"].items() if k != "scope"},
        }

    failures = check_floors({name: result["metrics"] for name, result in results.items()})
    return {
        "purpose": "CI regression guard on a corpus slice; not a reported benchmark",
        "corpus": SMOKE_CORPUS,
        "document_count": len(documents),
        "chunk_count": ingestion.chunk_count,
        "query_count": len(queries),
        "embedding_model": embedder.model_identity,
        "floors": FLOORS,
        "results": results,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    report = run_smoke()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
