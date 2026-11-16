import json

import pytest

from app.evaluation.compare import compare, paired_bootstrap


def test_bootstrap_is_deterministic_and_zero_for_identical_configs():
    values = [0.0, 1.0, 0.5, 1.0]

    assert paired_bootstrap(values, values) == {
        "mean_difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
    }
    first = paired_bootstrap([0, 0, 1, 0, 1], [1, 0, 1, 1, 1], samples=2000)
    assert first == paired_bootstrap([0, 0, 1, 0, 1], [1, 0, 1, 1, 1], samples=2000)
    assert first["mean_difference"] == 0.4
    assert first["ci95_low"] <= 0.4 <= first["ci95_high"]


def test_bootstrap_interval_excludes_zero_for_a_consistent_gain():
    result = paired_bootstrap([0.0] * 40, [1.0] * 30 + [0.0] * 10, samples=2000)

    assert result["ci95_low"] > 0


def _report(config, values, dataset="d", corpus="c"):
    return {
        "dataset": {"sha256": dataset},
        "corpus": {"sha256": corpus},
        "config": {"method": config},
        "environment": {"git_commit": "abc"},
        "evaluated_at": "2026-11-16",
        "retrieval": {
            "answerable_query_count": len(values),
            "metrics": {},
            "metrics_by_category": {},
            "latency": {"scope": "x", "p50_ms": 1.0},
            "cases": [
                {"id": f"q{i}", "relevant_source_ids": ["a"], "recall@1": v, "recall@3": v,
                 "recall@5": v, "reciprocal_rank": v, "ndcg@5": v}
                for i, v in enumerate(values)
            ] + [{"id": "u", "relevant_source_ids": []}],
        },
    }


def test_compare_reads_results_and_rejects_mismatched_datasets(tmp_path):
    for name, report in {
        "base": _report("base", [0, 0, 1, 1]),
        "better": _report("better", [1, 1, 1, 1]),
        "other": _report("other", [1, 1, 1, 1], dataset="changed"),
    }.items():
        (tmp_path / f"results-v2-{name}.json").write_text(json.dumps(report))

    comparison = compare(["base", "better"], tmp_path)

    assert comparison["differences_vs_baseline"]["better"]["mrr@10"]["mean_difference"] == 0.5
    assert comparison["configs"]["base"]["latency"] == {"p50_ms": 1.0}
    with pytest.raises(ValueError, match="different dataset"):
        compare(["base", "other"], tmp_path)
