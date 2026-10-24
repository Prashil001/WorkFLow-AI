
import pytest

from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)
from app.workflow_engine.validation import (
    PlanValidationError,
    validate_plan,
)
from app.workflow_engine.validation import get_execution_order


def make_task(
    task_id: str,
    depends_on: list[str] | None = None,
) -> PlannedTask:
    return PlannedTask(
        task_id=task_id,
        task_type="retrieve_documents",
        description=f"Test task {task_id}",
        depends_on=(
            depends_on
            if depends_on is not None
            else []
        ),
    )


def make_plan(
    *tasks: PlannedTask,
    missing_information: list[str] | None = None,
) -> WorkflowPlan:
    return WorkflowPlan(
        summary="Test workflow",
        missing_information=(
            missing_information
            if missing_information is not None
            else []
        ),
        tasks=list(tasks),
    )


def test_accepts_valid_plan():
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ["T1"]),
    )

    result = validate_plan(plan)

    assert result is plan

def test_rejects_duplicate_task_ids():
    plan = make_plan(
        make_task("T1"),
        make_task("T1"),
    )

    with pytest.raises(
        PlanValidationError,
        match="Duplicate task IDs",
    ):
        validate_plan(plan)

def test_rejects_unknown_dependency():
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ["T99"]),
    )

    with pytest.raises(
        PlanValidationError,
        match="unknown task T99",
    ):
        validate_plan(plan)

def test_rejects_self_dependency():
    plan = make_plan(
        make_task("T1", ["T1"]),
    )

    with pytest.raises(
        PlanValidationError,
        match="cannot depend on itself",
    ):
        validate_plan(plan)

def test_rejects_duplicate_dependencies():
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ["T1", "T1"]),
    )

    with pytest.raises(
        PlanValidationError,
        match="duplicate dependency",
    ):
        validate_plan(plan)

def test_rejects_blank_task_id():
    plan = make_plan(
        make_task("   "),
    )

    with pytest.raises(
        PlanValidationError,
        match="Task IDs cannot be empty",
    ):
        validate_plan(plan)

def test_accepts_clarification_only_plan():
    plan = make_plan(
        missing_information=[
            "Which client should we prepare "
            "the proposal for?"
        ],
    )

    result = validate_plan(plan)

    assert result is plan
    assert result.tasks == []

def test_rejects_empty_plan_without_questions():
    plan = make_plan()

    with pytest.raises(
        PlanValidationError,
        match="Plan must contain tasks",
    ):
        validate_plan(plan)


def test_execution_order():
    plan = make_plan(
        make_task("T3", ["T1", "T2"]),
        make_task("T2", ["T1"]),
        make_task("T1"),
    )

    order = get_execution_order(plan)

    assert order == ["T1", "T2", "T3"]

def test_rejects_two_task_cycle():
    plan = make_plan(
        make_task("T1", ["T2"]),
        make_task("T2", ["T1"]),
    )

    with pytest.raises(
        PlanValidationError,
        match="Circular dependency",
    ):
        get_execution_order(plan)

def test_rejects_three_task_cycle():
    plan = make_plan(
        make_task("T1", ["T3"]),
        make_task("T2", ["T1"]),
        make_task("T3", ["T2"]),
    )

    with pytest.raises(
        PlanValidationError,
        match="Circular dependency",
    ):
        get_execution_order(plan)

def test_independent_tasks():
    plan = make_plan(
        make_task("T1"),
        make_task("T2"),
        make_task("T3"),
    )

    order = get_execution_order(plan)

    assert order == ["T1", "T2", "T3"]

def test_clarification_only_order():
    plan = make_plan(
        missing_information=[
            "Which client should we prepare "
            "the proposal for?"
        ],
    )

    order = get_execution_order(plan)

    assert order == []