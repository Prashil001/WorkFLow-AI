import pytest

from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)
from app.workflow_engine.validation import PlanValidationError


def test_ready_plan_executes_in_dependency_order_and_stores_results():
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare proposal",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Draft the proposal",
                    depends_on=["T1"],
                ),
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find pricing",
                    depends_on=[],
                ),
            ],
        )

    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        source_ids = sorted({
            source_id
            for dependency in dependency_results.values()
            for source_id in dependency.source_ids
        })
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
            source_ids=source_ids,
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner,
        task_runner=recording_runner,
    )

    result = graph.invoke({
        "original_request": "Prepare a proposal"
    })

    assert result["planning_outcome"] == "READY"
    assert executed_ids == ["T1", "T2"]
    assert [item["task_id"] for item in result["execution_results"]] == [
        "T1",
        "T2",
    ]
    assert [item["status"] for item in result["execution_results"]] == [
        "SIMULATED",
        "SIMULATED",
    ]


def test_missing_information_plan_does_not_execute_tasks():
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Need client details",
            missing_information=["Which client is this for?"],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find pricing",
                    depends_on=[],
                ),
            ],
        )

    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner,
        task_runner=recording_runner,
    )

    result = graph.invoke({
        "original_request": "Prepare a proposal"
    })

    assert result["planning_outcome"] == "NEEDS_INPUT"
    assert executed_ids == []
    assert "execution_results" not in result


def test_invalid_plan_does_not_execute_tasks():
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Circular plan",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find pricing",
                    depends_on=["T2"],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Draft proposal",
                    depends_on=["T1"],
                ),
            ],
        )

    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner,
        task_runner=recording_runner,
    )

    with pytest.raises(PlanValidationError, match="Circular dependency"):
        graph.invoke({
            "original_request": "Prepare a proposal"
        })

    assert executed_ids == []


def test_failed_task_skips_dependent_in_graph():
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare proposal and check calendar",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find pricing",
                    depends_on=[],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Draft proposal",
                    depends_on=["T1"],
                ),
                PlannedTask(
                    task_id="T3",
                    task_type="check_calendar",
                    description="Check availability",
                    depends_on=[],
                ),
            ],
        )

    executed_ids = []

    def failing_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        if task.task_id == "T1":
            raise RuntimeError("Pricing source unavailable")
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner,
        task_runner=failing_runner,
    )

    result = graph.invoke({
        "original_request": "Prepare a proposal"
    })

    results_by_id = {
        item["task_id"]: item
        for item in result["execution_results"]
    }

    assert executed_ids == ["T1", "T3"]
    assert results_by_id["T1"]["status"] == "FAILED"
    assert results_by_id["T2"]["status"] == "SKIPPED"
    assert results_by_id["T3"]["status"] == "SIMULATED"
