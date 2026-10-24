from app.workflow_engine.schemas import WorkflowPlan

from collections import deque

class PlanValidationError(ValueError):
    """Raised when a task plan is logically invalid"""
    pass

def validate_plan(plan: WorkflowPlan) -> WorkflowPlan:

    # Rule 1: An empty plan must explain
    # should explain what information is missing
    if not plan.tasks:
        has_questions = any(
            questions.strip()
            for questions in plan.missing_information
        )

        if not has_questions:
            raise PlanValidationError("Plan must contain tasks or request missing information")

        return plan

    # Rule 2: Every task must have a valid ID
    task_ids = [task.task_id for task in plan.tasks]

    for task_id in task_ids:
        if not task_id or any(
            char.isspace() for char in task_id
        ):
            raise PlanValidationError(
                "Task IDs cannot be empty or contain whitespace"
            )

    # Rule 3: Task ID must be unique
    if len(task_ids) != len(set(task_ids)):
        raise PlanValidationError(
            "Duplicate task IDs detected"
        )
    known_task_ids = set(task_ids)

    # Rule 4: Validate each task's dependencies
    for task in plan.tasks:
        seen_dependencies = set()
        for dependency in task.depends_on:
            if dependency in seen_dependencies:
                raise PlanValidationError(
                    f"Task {task.task_id} has duplicate dependency {dependency}"
                )
            seen_dependencies.add(dependency)

            if dependency == task.task_id:
                raise PlanValidationError(
                    f"Task {task.task_id} "
                    "cannot depend on itself"                    
                )

            if dependency not in known_task_ids:
                raise PlanValidationError(
                    f"Task {task.task_id} depends "
                    f"on unknown task {dependency}"                    
                )
    return plan



def get_execution_order(
    plan: WorkflowPlan,
) -> list[str]:

    # First run our existing validation checks.
    validate_plan(plan)

    # Count how many dependencies each task has.
    remaining_dependencies = {
        task.task_id: len(task.depends_on)
        for task in plan.tasks
    }

    # Store which tasks depend on each task.
    dependents = {
        task.task_id: []
        for task in plan.tasks
    }

    for task in plan.tasks:
        for dependency in task.depends_on:
            dependents[dependency].append(
                task.task_id
            )

    # Find tasks with no dependencies.
    ready = deque(
        task.task_id
        for task in plan.tasks
        if remaining_dependencies[task.task_id] == 0
    )

    execution_order = []

    while ready:
        task_id = ready.popleft()

        execution_order.append(task_id)

        # Update tasks that were waiting for this task.
        for dependent_id in dependents[task_id]:
            remaining_dependencies[dependent_id] -= 1

            if remaining_dependencies[dependent_id] == 0:
                ready.append(dependent_id)

    # If tasks remain unordered, a cycle exists.
    if len(execution_order) != len(plan.tasks):
        raise PlanValidationError(
            "Circular dependency detected"
        )

    return execution_order
