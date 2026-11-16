"""Evaluate retrieval and grounded generation on the v2 RAG dataset.

    uv run python -m app.evaluation.run_v2 --config vector
    uv run python -m app.evaluation.run_v2 --config vector --skip-generation

Retrieval metrics are document-level (chunks are collapsed to their source page)
and computed over answerable queries. Generation metrics use an LLM judge and
deterministic citation checks. Results go to evaluation/results-v2-<config>.json.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset_checks import (
    DATASET_V2_PATH,
    find_dataset_problems,
    load_dataset_v2,
)
from app.evaluation.generation import (
    DEFAULT_JUDGE_MODEL,
    GENERATOR_INSTRUCTIONS,
    JUDGE_INSTRUCTIONS,
    PROMPT_VERSION,
    REASONING_EFFORT,
    AnswerGenerator,
    AnswerJudge,
    check_citations,
    is_abstention,
)
from app.evaluation.metrics import (
    average,
    latency_summary,
    ndcg_at_k,
    rank_documents,
    recall_at_k,
    reciprocal_rank,
)
from app.workflow_engine.rag import Retriever, RetrievedChunk


CANDIDATE_CHUNKS = 30  # chunks requested per query; collapsed to unique pages
RANKED_DOCUMENTS = 10  # pages kept for MRR@10
CONTEXT_CHUNKS = 5  # chunks handed to the generator
RECALL_KS = (1, 3, 5)
NDCG_K = 5
WARMUP_QUERIES = 3
RESULTS_DIR = Path("evaluation")


@dataclass
class RetrieverConfig:
    name: str
    retriever: Retriever
    description: dict[str, Any] = field(default_factory=dict)


def build_vector_config(corpus: str | None = None, embedder: Any = None) -> RetrieverConfig:
    from app.rag.corpus import DOCS_CORPUS
    from app.rag.pgvector_retriever import PgVectorRetriever
    from app.workflow_engine.rag import FastEmbedTextEmbedder

    corpus = corpus or DOCS_CORPUS
    embedder = embedder or FastEmbedTextEmbedder()
    return RetrieverConfig(
        name="vector",
        retriever=PgVectorRetriever(embedder, corpus=corpus, minimum_score=0.0),
        description={
            "method": "dense cosine similarity over pgvector HNSW (m=16, ef_construction=64)",
            "embedding_model": embedder.model_identity,
            "corpus": corpus,
        },
    )


def build_hybrid_config(corpus: str | None = None, embedder: Any = None) -> RetrieverConfig:
    from app.rag.hybrid import RRF_K, HybridRetriever, PgFullTextRetriever
    from app.workflow_engine.rag import MAX_RETRIEVAL_LIMIT

    vector = build_vector_config(corpus, embedder)
    return RetrieverConfig(
        name="hybrid",
        retriever=HybridRetriever(
            vector.retriever,
            PgFullTextRetriever(corpus=vector.description["corpus"]),
        ),
        description={
            **vector.description,
            "method": "Reciprocal Rank Fusion of dense pgvector and Postgres full-text "
            "(OR of query terms, ts_rank_cd, GIN index)",
            "rrf_k": RRF_K,
            "candidates_per_retriever": MAX_RETRIEVAL_LIMIT,
        },
    )


def _build_reranked_config(name: str, model_name: str) -> RetrieverConfig:
    from app.rag.hybrid import RERANK_DEPTH, FastEmbedCrossEncoder, RerankingRetriever

    hybrid = build_hybrid_config()
    return RetrieverConfig(
        name=name,
        retriever=RerankingRetriever(hybrid.retriever, FastEmbedCrossEncoder(model_name)),
        description={
            **hybrid.description,
            "method": f"{hybrid.description['method']}, then cross-encoder reranking "
            f"of the top {RERANK_DEPTH} chunks",
            "reranker_model": model_name,
            "rerank_depth": RERANK_DEPTH,
        },
    )


CONFIG_BUILDERS: dict[str, Callable[[], RetrieverConfig]] = {
    "vector": build_vector_config,
    "hybrid": build_hybrid_config,
    "hybrid-rerank": lambda: _build_reranked_config(
        "hybrid-rerank", "BAAI/bge-reranker-base"
    ),
    "hybrid-rerank-minilm": lambda: _build_reranked_config(
        "hybrid-rerank-minilm", "Xenova/ms-marco-MiniLM-L-6-v2"
    ),
}


def evaluate_retrieval(
    queries: Sequence[dict[str, Any]],
    retriever: Retriever,
    *,
    clock: Callable[[], float] = time.perf_counter,
    warmup: int = WARMUP_QUERIES,
) -> tuple[dict[str, Any], dict[str, list[RetrievedChunk]]]:
    """Return (retrieval report, ranked chunks per query id)."""
    for query in queries[:warmup]:
        retriever.retrieve(query["question"], limit=CANDIDATE_CHUNKS)

    per_query = []
    chunks_by_query: dict[str, list[RetrievedChunk]] = {}
    latencies: list[float] = []
    for query in queries:
        started = clock()
        chunks = retriever.retrieve(query["question"], limit=CANDIDATE_CHUNKS)
        latencies.append((clock() - started) * 1000)
        chunks_by_query[query["id"]] = chunks

        ranked = rank_documents(chunks)[:RANKED_DOCUMENTS]
        relevant = set(query["relevant_source_ids"])
        record: dict[str, Any] = {
            "id": query["id"],
            "category": query["category"],
            "relevant_source_ids": query["relevant_source_ids"],
            "retrieved_source_ids": ranked[:5],
            "top_chunk_score": round(chunks[0].score, 4) if chunks else None,
        }
        if relevant:
            record.update({
                **{f"recall@{k}": recall_at_k(ranked, relevant, k) for k in RECALL_KS},
                "reciprocal_rank": reciprocal_rank(ranked, relevant, RANKED_DOCUMENTS),
                f"ndcg@{NDCG_K}": round(ndcg_at_k(ranked, relevant, NDCG_K), 4),
                "first_relevant_rank": next(
                    (rank for rank, doc in enumerate(ranked, start=1) if doc in relevant),
                    None,
                ),
            })
        per_query.append(record)

    answerable = [record for record in per_query if record["relevant_source_ids"]]
    report = {
        "query_count": len(per_query),
        "answerable_query_count": len(answerable),
        "metrics": _retrieval_metrics(answerable),
        "metrics_by_category": {
            category: _retrieval_metrics(
                [record for record in answerable if record["category"] == category]
            )
            for category in sorted({record["category"] for record in answerable})
        },
        "latency": {
            "scope": "retriever.retrieve() wall time incl. query embedding; "
            f"{warmup} warm-up queries excluded",
            **latency_summary(latencies),
        },
        "top_chunk_score_mean": {
            "answerable": average(
                record["top_chunk_score"] for record in answerable
                if record["top_chunk_score"] is not None
            ),
            "unanswerable": average(
                record["top_chunk_score"] for record in per_query
                if not record["relevant_source_ids"] and record["top_chunk_score"] is not None
            ),
        },
        "cases": per_query,
    }
    return report, chunks_by_query


def _retrieval_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"query_count": 0}
    return {
        "query_count": len(records),
        **{f"recall@{k}": average(r[f"recall@{k}"] for r in records) for k in RECALL_KS},
        "mrr@10": average(r["reciprocal_rank"] for r in records),
        f"ndcg@{NDCG_K}": average(r[f"ndcg@{NDCG_K}"] for r in records),
    }


def evaluate_generation(
    queries: Sequence[dict[str, Any]],
    chunks_by_query: dict[str, list[RetrievedChunk]],
    generator: AnswerGenerator,
    judge: AnswerJudge,
    *,
    previous_cases: dict[str, dict[str, Any]] | None = None,
    pace_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    previous_cases = previous_cases or {}
    cases = []
    reused = 0
    for query in queries:
        context = chunks_by_query[query["id"]][:CONTEXT_CHUNKS]
        relevant = set(query["relevant_source_ids"])
        previous = previous_cases.get(query["id"])
        if (
            previous is not None
            and not previous.get("generation_error")
            and not previous.get("judge_error")
            and previous["context_source_ids"] == [chunk.source_id for chunk in context]
        ):
            # Same question, same retrieved context, finished without errors.
            cases.append(previous)
            reused += 1
            continue
        case: dict[str, Any] = {
            "id": query["id"],
            "category": query["category"],
            "answerable": bool(relevant),
            "context_source_ids": [chunk.source_id for chunk in context],
        }
        try:
            answer = generator(query["question"], context)
        except Exception as exc:
            # Recorded and counted (see generation_error_count); never dropped.
            case.update({
                "answer": None,
                "abstained": False,
                "generation_error": f"{type(exc).__name__}: {exc}",
            })
            cases.append(case)
            continue

        abstained = is_abstention(answer)
        citations = check_citations(answer, {c.source_id for c in context}, relevant)
        case.update({
            "answer": answer,
            "abstained": abstained,
            "citations": asdict(citations),
        })
        if not abstained:
            try:
                verdict = judge(query["question"], query["reference_answer"], context, answer)
            except Exception as exc:
                case["judge_error"] = f"{type(exc).__name__}: {exc}"
            else:
                supported = sum(claim.supported for claim in verdict.claims)
                case["judge"] = verdict.model_dump()
                case["faithfulness"] = (
                    round(supported / len(verdict.claims), 4) if verdict.claims else None
                )
        cases.append(case)
        if pace_seconds:
            sleep(pace_seconds)

    return {
        "reused_case_count": reused,
        "metrics": _generation_metrics(cases),
        "cases": cases,
    }


def _generation_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    # "answered" = produced a non-abstaining answer (generation errors excluded).
    answered = [c for c in cases if c["answer"] is not None and not c["abstained"]]
    answered_answerable = [c for c in answered if c["answerable"]]
    judged_answerable = [c for c in answered_answerable if "judge" in c]

    def judge_score(case: dict[str, Any], name: str) -> float:
        # Abstentions, generation errors, and judge errors on answerable
        # questions count as the lowest score.
        return case["judge"][name] if "judge" in case else 1

    return {
        "generation_error_count": sum("generation_error" in c for c in cases),
        "judge_error_count": sum("judge_error" in c for c in cases),
        "error_handling": "generation or judge errors are kept in the cases; on answerable "
        "queries they score 1 for correctness and count as not abstaining",
        "answered_count": len(answered),
        "faithfulness": {
            "definition": "share of the answer's claims supported by the retrieved context "
            "(LLM judge), averaged over non-abstaining answers with at least one claim",
            "mean": average(c["faithfulness"] for c in answered if c.get("faithfulness") is not None),
            "fully_grounded_rate": average(
                float(c["faithfulness"] == 1.0)
                for c in answered if c.get("faithfulness") is not None
            ),
        },
        "answer_relevance": {
            "definition": "judge score 1-5 over answerable queries that were answered",
            "mean": average(c["judge"]["answer_relevance"] for c in judged_answerable),
        },
        "answer_correctness": {
            "definition": "judge score 1-5 vs reference answer over all answerable queries; "
            "abstentions score 1",
            "mean": average(judge_score(c, "answer_correctness") for c in answerable),
            "score_4_or_5_rate": average(
                float(judge_score(c, "answer_correctness") >= 4) for c in answerable
            ),
        },
        "abstention": {
            "unanswerable_abstention_rate": average(float(c["abstained"]) for c in unanswerable),
            "answerable_false_abstention_rate": average(float(c["abstained"]) for c in answerable),
        },
        "citation_integrity": {
            "valid_citation_rate": average(float(c["citations"]["valid"]) for c in answered),
            "uncited_answer_rate": average(
                float(not c["citations"]["has_citation"]) for c in answered
            ),
            "cites_relevant_source_rate": average(
                float(bool(c["citations"]["cites_relevant"])) for c in answered_answerable
            ),
            "definitions": {
                "valid_citation_rate": "answers whose cited IDs all appear in the context given",
                "uncited_answer_rate": "non-abstaining answers with no citation",
                "cites_relevant_source_rate": "answered answerable queries citing a gold page",
            },
        },
    }


def run_evaluation(
    config: RetrieverConfig,
    *,
    dataset_path: Path = DATASET_V2_PATH,
    generator: AnswerGenerator | None = None,
    judge: AnswerJudge | None = None,
    generation_details: dict[str, Any] | None = None,
    previous_report: dict[str, Any] | None = None,
    pace_seconds: float = 0.0,
) -> dict[str, Any]:
    dataset = load_dataset_v2(dataset_path)
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    problems = find_dataset_problems(dataset)
    if problems:
        raise ValueError(f"Dataset failed verification: {problems[:3]}")

    queries = dataset["queries"]
    retrieval, chunks_by_query = evaluate_retrieval(queries, config.retriever)
    generation = None
    if generator is not None and judge is not None:
        generation = {
            **(generation_details or {}),
            "context_chunks": CONTEXT_CHUNKS,
            **evaluate_generation(
                queries,
                chunks_by_query,
                generator,
                judge,
                previous_cases=_resumable_cases(
                    previous_report, dataset_sha, generation_details or {}
                ),
                pace_seconds=pace_seconds,
            ),
        }

    return {
        "dataset": {
            "name": dataset["dataset_name"],
            "version": dataset["version"],
            "sha256": dataset_sha,
            "query_count": len(queries),
            "category_counts": _category_counts(queries),
        },
        "corpus": _corpus_details(),
        "config": {"name": config.name, **config.description,
                   "candidate_chunks": CANDIDATE_CHUNKS},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "retrieval": retrieval,
        "generation": generation,
        "limitations": [
            "80 hand-verified queries over one 227-page corpus: differences of a few "
            "points between configs are within noise for a set this size.",
            "Relevance labels are binary and document-level; they were audited against "
            "dense and full-text pools, so pages neither system retrieved may be unlabeled.",
            "Generation scores come from an LLM judge, not human raters.",
            "Latency was measured on one developer machine with a local PostgreSQL in "
            "Docker; it includes CPU query embedding.",
        ],
    }


_RESUME_KEYS = ("generator_model", "judge_model", "prompt_sha256", "reasoning_effort")


def _resumable_cases(
    previous_report: dict[str, Any] | None,
    dataset_sha: str,
    generation_details: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Cases from an earlier run that may be reused: same dataset, models, prompts."""
    if not previous_report or not previous_report.get("generation"):
        return {}
    previous_generation = previous_report["generation"]
    if previous_report["dataset"]["sha256"] != dataset_sha:
        return {}
    if any(previous_generation.get(key) != generation_details.get(key) for key in _RESUME_KEYS):
        return {}
    return {case["id"]: case for case in previous_generation["cases"]}


def _category_counts(queries: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        counts[query["category"]] = counts.get(query["category"], 0) + 1
    return counts


def _corpus_details() -> dict[str, Any]:
    from app.rag.corpus import DOCS_CORPUS_PATH, load_docs_corpus_file

    corpus = load_docs_corpus_file()
    return {
        "name": corpus["corpus"],
        "version": corpus["version"],
        "document_count": corpus["document_count"],
        "sha256": hashlib.sha256(DOCS_CORPUS_PATH.read_bytes()).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        details["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        details["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        from sqlalchemy import text

        from app.db.database import engine

        with engine.connect() as connection:
            details["postgres"] = connection.execute(text("SHOW server_version")).scalar_one()
            details["pgvector"] = connection.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )).scalar_one_or_none()
    except Exception:
        pass
    return details


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", choices=sorted(CONFIG_BUILDERS), required=True)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--pace", type=float, default=0.0,
                        help="Seconds to wait between queries (provider rate limits).")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse error-free generation cases from the existing output file when the "
        "dataset, models, prompts, and retrieved context are unchanged.",
    )
    args = parser.parse_args()
    output = args.output or RESULTS_DIR / f"results-v2-{args.config}.json"

    config = CONFIG_BUILDERS[args.config]()
    generator = judge = None
    generation_details = None
    if not args.skip_generation:
        from app.core.config import get_settings
        from app.evaluation.generation import GroqAnswerGenerator, GroqAnswerJudge

        generator = GroqAnswerGenerator(get_settings().groq_model)
        judge = GroqAnswerJudge(args.judge_model)
        generation_details = {
            "generator_model": generator.model_name,
            "judge_model": judge.model_name,
            "temperature": 0,
            "reasoning_effort": REASONING_EFFORT,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(
                (GENERATOR_INSTRUCTIONS + JUDGE_INSTRUCTIONS).encode("utf-8")
            ).hexdigest(),
        }

    previous_report = None
    if args.resume and output.exists():
        previous_report = json.loads(output.read_text(encoding="utf-8"))

    report = run_evaluation(
        config,
        generator=generator,
        judge=judge,
        generation_details=generation_details,
        previous_report=previous_report,
        pace_seconds=args.pace,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = {"retrieval": report["retrieval"]["metrics"],
               "latency": report["retrieval"]["latency"]}
    if report["generation"]:
        summary["generation"] = report["generation"]["metrics"]
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
