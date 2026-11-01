import json
from collections.abc import Callable, Mapping

from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)
from app.workflow_engine.mock_tools import run_read_only_tool
from app.workflow_engine.drafting import generate_grounded_draft
from app.workflow_engine.mock_data import load_pricing_documents_by_ids


DependencyResults = Mapping[str, TaskExecutionResult]
DraftGenerator = Callable[
    [PlannedTask, list[dict[str, str]], DependencyResults],
    str,
]
TaskRunner = Callable[
    [PlannedTask, dict[str, TaskExecutionResult]],
    TaskExecutionResult,
]


def execute_task(
    task: PlannedTask,
    dependency_results: dict[str, TaskExecutionResult] | None = None,
    *,
    retriever=None,
    draft_generator: DraftGenerator = generate_grounded_draft,
) -> TaskExecutionResult:
    """Execute one task using local tools and grounded draft generation."""
    dependency_results = dependency_results or {}
    tool_result = run_read_only_tool(task, retriever=retriever)
    if tool_result is not None:
        tool_payload = json.loads(tool_result)
        source_ids = sorted({
            match["source_id"]
            for match in tool_payload.get("matches", [])
            if "source_id" in match
        })
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=tool_result,
            source_ids=source_ids,
        )

    if task.task_type in {
        "draft_proposal",
        "draft_email",
        "create_follow_up",
    }:
        source_ids = {
            source_id
            for dependency in dependency_results.values()
            if dependency.status == "SIMULATED"
            for source_id in dependency.source_ids
        }
        if not source_ids:
            return TaskExecutionResult(
                task_id=task.task_id,
                status="FAILED",
                error="No retrieved source documents are available for a grounded draft",
            )

        source_documents = load_pricing_documents_by_ids(source_ids)
        if not source_documents:
            return TaskExecutionResult(
                task_id=task.task_id,
                status="FAILED",
                error="Retrieved source IDs did not match available documents",
            )
        draft = draft_generator(
            task,
            source_documents,
            dependency_results,
        )
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=draft,
            source_ids=sorted(source_ids),
        )

    return TaskExecutionResult(
        task_id=task.task_id,
        status="SIMULATED",
        result=(
            f"Simulated {task.task_type} task: {task.description}"
        ),
    )


def execute_plan(
    plan: WorkflowPlan,
    execution_order: list[str],
    task_runner: TaskRunner = execute_task,
) -> list[TaskExecutionResult]:
    """Execute tasks in validated order, skipping blocked dependents."""
    if plan.missing_information or not plan.tasks:
        return []

    tasks_by_id = {
        task.task_id: task
        for task in plan.tasks
    }
    results_by_id: dict[str, TaskExecutionResult] = {}
    results: list[TaskExecutionResult] = []

    for task_id in execution_order:
        task = tasks_by_id[task_id]
        blocked_dependency = next(
            (
                dependency_id
                for dependency_id in task.depends_on
                if (
                    dependency_id not in results_by_id
                    or results_by_id[dependency_id].status != "SIMULATED"
                )
            ),
            None,
        )

        if blocked_dependency is not None:
            task_result = TaskExecutionResult(
                task_id=task.task_id,
                status="SKIPPED",
                error=(
                    f"Dependency {blocked_dependency} did not complete successfully"
                ),
            )
        else:
            try:
                dependency_results = {
                    dependency_id: results_by_id[dependency_id]
                    for dependency_id in task.depends_on
                    if dependency_id in results_by_id
                }
                task_result = task_runner(task, dependency_results)
            except Exception as exc:
                task_result = TaskExecutionResult(
                    task_id=task.task_id,
                    status="FAILED",
                    error=str(exc) or type(exc).__name__,
                )

            if task_result.task_id != task.task_id:
                task_result = TaskExecutionResult(
                    task_id=task.task_id,
                    status="FAILED",
                    error="Task runner returned a result for a different task",
                )

        results_by_id[task_id] = task_result
        results.append(task_result)

    return results
