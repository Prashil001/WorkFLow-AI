from app.workflow_engine.executor import execute_plan, execute_task
from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)


def make_plan(tasks, missing_information=None):
    return WorkflowPlan(
        summary="Test plan",
        missing_information=missing_information or [],
        tasks=tasks,
    )


def test_draft_task_without_retrieved_sources_fails_safely():
    task = PlannedTask(
        task_id="T1",
        task_type="draft_email",
        description="Draft a follow-up email",
        depends_on=[],
    )

    result = execute_task(task)

    assert result == TaskExecutionResult(
        task_id="T1",
        status="FAILED",
        error="No retrieved source documents are available for a grounded draft",
    )


def test_execute_plan_follows_validated_order():
    plan = make_plan([
        PlannedTask(
            task_id="T2",
            task_type="draft_proposal",
            description="Draft proposal",
            depends_on=["T1"],
        ),
        PlannedTask(
            task_id="T1",
            task_type="retrieve_documents",
            description="Find pricing",
            depends_on=[],
        ),
    ])
    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    results = execute_plan(
        plan,
        execution_order=["T1", "T2"],
        task_runner=recording_runner,
    )

    assert executed_ids == ["T1", "T2"]
    assert [result.status for result in results] == [
        "SIMULATED",
        "SIMULATED",
    ]


def test_failed_task_skips_dependents_and_continues_independent_tasks():
    plan = make_plan([
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
    ])
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

    results = execute_plan(
        plan,
        execution_order=["T1", "T2", "T3"],
        task_runner=failing_runner,
    )

    assert executed_ids == ["T1", "T3"]
    assert [result.status for result in results] == [
        "FAILED",
        "SKIPPED",
        "SIMULATED",
    ]
    assert results[0].error == "Pricing source unavailable"
    assert results[1].error == "Dependency T1 did not complete successfully"


def test_plan_with_missing_information_does_not_execute():
    plan = make_plan(
        [
            PlannedTask(
                task_id="T1",
                task_type="retrieve_documents",
                description="Find pricing",
                depends_on=[],
            ),
        ],
        missing_information=["Which client is this for?"],
    )
    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    results = execute_plan(
        plan,
        execution_order=["T1"],
        task_runner=recording_runner,
    )

    assert results == []
    assert executed_ids == []
