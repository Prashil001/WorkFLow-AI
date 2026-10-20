
import pytest
from pydantic import ValidationError

from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)


def test_valid_workflow_plan():
    plan = WorkflowPlan(
        summary="Prepare client proposal",
        missing_information=[],
        tasks=[
            PlannedTask(
                task_id="T1",
                task_type="retrieve_documents",
                description="Find pricing documents",
                depends_on=[],
            ),
            PlannedTask(
                task_id="T2",
                task_type="draft_proposal",
                description="Prepare proposal",
                depends_on=["T1"],
            ),
        ],
    )

    assert len(plan.tasks) == 2
    assert plan.tasks[1].depends_on == ["T1"]


def test_rejects_unknown_task_type():
    with pytest.raises(ValidationError):
        PlannedTask(
            task_id="T1",
            task_type="delete_all_emails",
            description="Delete emails",
            depends_on=[],
        )


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PlannedTask.model_validate({
            "task_id": "T1",
            "task_type": "retrieve_documents",
            "description": "Find pricing",
            "depends_on": [],
            "secret_command": "delete_database",
        })


def test_rejects_missing_fields():
    with pytest.raises(ValidationError):
        WorkflowPlan.model_validate({
            "summary": "Prepare a proposal",
            "tasks": [],
        })
