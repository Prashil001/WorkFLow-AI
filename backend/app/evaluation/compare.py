"""Compare saved v2 results files against a baseline with paired bootstrap CIs.

    uv run python -m app.evaluation.compare vector hybrid hybrid-rerank-minilm hybrid-rerank

The first config is the baseline. For each other config and metric, queries are
resampled with replacement (paired: the same queries for both configs) and the
95% percentile interval of the mean difference is reported. An interval that
excludes 0 is a difference this 65-query set can resolve; one that includes 0
is not distinguishable from noise.
"""

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.evaluation.run_v2 import RESULTS_DIR


METRICS = ("recall@1", "recall@3", "recall@5", "reciprocal_rank", "ndcg@5")
LABELS = {"reciprocal_rank": "mrr@10"}
BOOTSTRAP_SAMPLES = 10_000
SEED = 20261116


def load_results(config: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    return json.loads((results_dir / f"results-v2-{config}.json").read_text(encoding="utf-8"))


def per_query(report: dict[str, Any], metric: str) -> dict[str, float]:
    return {
        case["id"]: float(case[metric])
        for case in report["retrieval"]["cases"]
        if case["relevant_source_ids"]
    }


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples must be non-empty and the same length")
    differences = [c - b for b, c in zip(baseline, candidate)]
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "mean_difference": round(sum(differences) / count, 4),
        "ci95_low": round(means[int(0.025 * samples)], 4),
        "ci95_high": round(means[int(0.975 * samples) - 1], 4),
    }


def compare(configs: Sequence[str], results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    reports = {config: load_results(config, results_dir) for config in configs}
    baseline_name = configs[0]
    baseline = reports[baseline_name]

    for name, report in reports.items():
        if report["dataset"]["sha256"] != baseline["dataset"]["sha256"]:
            raise ValueError(f"{name} used a different dataset than {baseline_name}")
        if report["corpus"]["sha256"] != baseline["corpus"]["sha256"]:
            raise ValueError(f"{name} used a different corpus than {baseline_name}")

    rows = {}
    for name, report in reports.items():
        retrieval = report["retrieval"]
        rows[name] = {
            "description": report["config"].get("method"),
            "reranker_model": report["config"].get("reranker_model"),
            "git_commit": report["environment"].get("git_commit"),
            "evaluated_at": report["evaluated_at"],
            "metrics": retrieval["metrics"],
            "metrics_by_category": retrieval["metrics_by_category"],
            "latency": {k: v for k, v in retrieval["latency"].items() if k != "scope"},
        }

    deltas: dict[str, dict[str, Any]] = {}
    for name in configs[1:]:
        deltas[name] = {}
        for metric in METRICS:
            base_values = per_query(baseline, metric)
            candidate_values = per_query(reports[name], metric)
            ids = sorted(base_values)
            if sorted(candidate_values) != ids:
                raise ValueError(f"{name} and {baseline_name} cover different queries")
            deltas[name][LABELS.get(metric, metric)] = paired_bootstrap(
                [base_values[i] for i in ids],
                [candidate_values[i] for i in ids],
            )

    return {
        "baseline": baseline_name,
        "dataset_sha256": baseline["dataset"]["sha256"],
        "corpus_sha256": baseline["corpus"]["sha256"],
        "answerable_query_count": baseline["retrieval"]["answerable_query_count"],
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED, "interval": "95% percentile"},
        "configs": rows,
        "differences_vs_baseline": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="+", help="Baseline first.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "comparison-v2.json")
    args = parser.parse_args()
    if len(args.configs) < 2:
        parser.error("Give a baseline and at least one config to compare")

    comparison = compare(args.configs)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison["differences_vs_baseline"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
