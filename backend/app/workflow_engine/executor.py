from collections.abc import Callable

from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)


TaskRunner = Callable[[PlannedTask], TaskExecutionResult]


def execute_task(task: PlannedTask) -> TaskExecutionResult:
    """Return a deterministic mock result for one planned task."""
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
                task_result = task_runner(task)
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
