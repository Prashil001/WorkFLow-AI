"""Ranking and latency metrics for document-level retrieval evaluation."""

from collections.abc import Iterable, Sequence
from math import log2
from statistics import mean, quantiles

from app.workflow_engine.rag import RetrievedChunk


def rank_documents(chunks: Iterable[RetrievedChunk]) -> list[str]:
    """Collapse a ranked chunk list to a ranked list of unique source IDs."""
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_id not in seen:
            seen.add(chunk.source_id)
            ranked.append(chunk.source_id)
    return ranked


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant documents")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for position, source_id in enumerate(ranked[:k], start=1):
        if source_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        raise ValueError("nDCG is undefined without relevant documents")
    dcg = sum(
        1.0 / log2(position + 1)
        for position, source_id in enumerate(ranked[:k], start=1)
        if source_id in relevant
    )
    ideal = sum(1.0 / log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def latency_summary(milliseconds: Sequence[float]) -> dict[str, float | int]:
    if not milliseconds:
        return {"count": 0}
    if len(milliseconds) == 1:
        only = round(milliseconds[0], 2)
        return {"count": 1, "p50_ms": only, "p95_ms": only, "mean_ms": only, "max_ms": only}
    cut_points = quantiles(milliseconds, n=100, method="inclusive")
    return {
        "count": len(milliseconds),
        "p50_ms": round(cut_points[49], 2),
        "p95_ms": round(cut_points[94], 2),
        "mean_ms": round(mean(milliseconds), 2),
        "max_ms": round(max(milliseconds), 2),
    }


def average(values: Iterable[float]) -> float | None:
    collected = list(values)
    return round(mean(collected), 4) if collected else None
