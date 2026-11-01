"""Run repeatable, rule-based evaluations without calling an LLM."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.workflow_engine.executor import execute_plan
from app.workflow_engine.rag import (
    EMBEDDING_MODEL_NAME,
    get_default_retriever,
    retrieve_pricing_documents,
)
from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)
from app.workflow_engine.validation import (
    PlanValidationError,
    get_execution_order,
)


DATASET_PATH = Path(__file__).with_name("dataset.v1.json")
_SOURCE_CITATION = re.compile(r"\[source:([A-Za-z0-9_-]+)\]")


def load_dataset(path: Path = DATASET_PATH) -> tuple[dict[str, Any], str]:
    dataset_bytes = path.read_bytes()
    return json.loads(dataset_bytes), hashlib.sha256(dataset_bytes).hexdigest()


def evaluate_planning_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        plan = None
        order = None
        try:
            plan = WorkflowPlan.model_validate(case["plan"])
            order = get_execution_order(plan)
            outcome = "NEEDS_INPUT" if plan.missing_information else "READY"
        except (PlanValidationError, ValueError, TypeError, KeyError):
            outcome = "INVALID"

        observed_types = (
            [task.task_type for task in plan.tasks]
            if plan is not None else None
        )
        observed_dependencies = (
            {task.task_id: task.depends_on for task in plan.tasks}
            if plan is not None else None
        )
        passed = (
            outcome == case["expected_outcome"]
            and observed_types == case["expected_task_types"]
            and observed_dependencies == case["expected_dependencies"]
            and order == case["expected_order"]
        )
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_outcome": case["expected_outcome"],
            "observed_outcome": outcome,
            "expected_order": case["expected_order"],
            "observed_order": order,
        })
    return _suite_result(results)


def evaluate_execution_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        plan = WorkflowPlan.model_validate(case["plan"])
        order = get_execution_order(plan)
        failed_task_ids = set(case["failed_task_ids"])

        def runner(
            task: PlannedTask,
            dependency_results: dict[str, TaskExecutionResult],
        ) -> TaskExecutionResult:
            if task.task_id in failed_task_ids:
                raise RuntimeError("Evaluation fixture failure")
            inherited_sources = sorted({
                source_id
                for result in dependency_results.values()
                for source_id in result.source_ids
            })
            return TaskExecutionResult(
                task_id=task.task_id,
                status="SIMULATED",
                result=f"Evaluation output for {task.task_id}",
                source_ids=inherited_sources,
            )

        execution_results = execute_plan(
            plan,
            order,
            task_runner=runner,
        )
        observed_statuses = {
            result.task_id: result.status
            for result in execution_results
        }
        observed_order = [result.task_id for result in execution_results]
        passed = (
            observed_order == case["expected_order"]
            and observed_statuses == case["expected_statuses"]
        )
        results.append({
            "id": case["id"],
            "passed": passed,
            "observed_order": observed_order,
            "observed_statuses": observed_statuses,
        })
    return _suite_result(results)


def evaluate_retrieval_cases(
    cases: list[dict[str, Any]],
    *,
    retriever=None,
) -> dict[str, Any]:
    active_retriever = retriever or get_default_retriever()
    results = []
    positive_cases = 0
    top_1_hits = 0
    top_3_hits = 0
    precision_at_3_total = 0.0
    negative_cases = 0
    correct_empty = 0

    for case in cases:
        matches = retrieve_pricing_documents(
            case["query"],
            limit=3,
            retriever=active_retriever,
        )
        observed_source_ids = [match["source_id"] for match in matches]
        expected_source_id = case["expected_source_id"]
        if expected_source_id is None:
            negative_cases += 1
            case_passed = not observed_source_ids
            correct_empty += int(case_passed)
            top_1_hit = None
            top_3_hit = None
        else:
            positive_cases += 1
            top_1_hit = bool(
                observed_source_ids
                and observed_source_ids[0] == expected_source_id
            )
            top_3_hit = expected_source_id in observed_source_ids[:3]
            top_1_hits += int(top_1_hit)
            top_3_hits += int(top_3_hit)
            if top_3_hit:
                precision_at_3_total += 1 / min(3, len(observed_source_ids))
            case_passed = top_3_hit

        results.append({
            "id": case["id"],
            "passed": case_passed,
            "expected_source_id": expected_source_id,
            "observed_source_ids": observed_source_ids,
            "top_1_hit": top_1_hit,
            "top_3_hit": top_3_hit,
        })

    return {
        **_suite_result(results),
        "metrics": {
            "positive_query_count": positive_cases,
            "hit_at_1": _rate(top_1_hits, positive_cases),
            "hit_at_3": _rate(top_3_hits, positive_cases),
            "precision_at_3": _rate(precision_at_3_total, positive_cases),
            "negative_query_count": negative_cases,
            "correct_empty_rate": _rate(correct_empty, negative_cases),
        },
    }


def evaluate_grounding_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        cited_source_ids = set(_SOURCE_CITATION.findall(case["draft"]))
        allowed_source_ids = set(case["allowed_source_ids"])
        required_source_ids = set(case["required_source_ids"])
        observed_integrity = (
            cited_source_ids.issubset(allowed_source_ids)
            and required_source_ids.issubset(cited_source_ids)
        )
        passed = observed_integrity == case["expected_integrity"]
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_integrity": case["expected_integrity"],
            "observed_integrity": observed_integrity,
            "cited_source_ids": sorted(cited_source_ids),
        })
    return _suite_result(results)


def evaluate_dataset(
    *,
    dataset_path: Path = DATASET_PATH,
    retriever=None,
) -> dict[str, Any]:
    dataset, dataset_hash = load_dataset(dataset_path)
    suites = {
        "planning": evaluate_planning_cases(dataset["planning_cases"]),
        "execution": evaluate_execution_cases(dataset["execution_cases"]),
        "retrieval": evaluate_retrieval_cases(
            dataset["retrieval_cases"],
            retriever=retriever,
        ),
        "grounding": evaluate_grounding_cases(dataset["grounding_cases"]),
    }
    total_cases = sum(suite["case_count"] for suite in suites.values())
    total_passed = sum(suite["passed_count"] for suite in suites.values())
    return {
        "dataset": {
            "name": dataset["dataset_name"],
            "version": dataset["version"],
            "sha256": dataset_hash,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "summary": {
            "passed_cases": total_passed,
            "total_cases": total_cases,
            "pass_rate": _rate(total_passed, total_cases),
            "all_passed": total_passed == total_cases,
        },
        "suites": suites,
        "limitations": [
            "Planning cases validate labeled plans and do not measure live LLM planner quality.",
            "Grounding checks citation ID integrity, not the truth of every generated sentence.",
            "The retrieval set is small and synthetic; its scores are a regression signal, not a broad benchmark.",
        ],
    }


def _suite_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed_count = sum(bool(result["passed"]) for result in results)
    return {
        "case_count": len(results),
        "passed_count": passed_count,
        "pass_rate": _rate(passed_count, len(results)),
        "all_passed": passed_count == len(results),
        "cases": results,
    }


def _rate(numerator: int | float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON results file.",
    )
    args = parser.parse_args()

    report = evaluate_dataset()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["summary"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
