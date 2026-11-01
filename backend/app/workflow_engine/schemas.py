from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal[
    "retrieve_documents",
    "check_calendar",
    "draft_proposal",
    "draft_email",
    "create_follow_up",
]

TaskExecutionStatus = Literal[
    "SIMULATED",
    "FAILED",
    "SKIPPED",
]

class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        description="Unique ID within this plan, e.g. T1"
    )

    task_type: TaskType = Field(
        description="Type of operation being proposed"
    )

    description: str = Field(
        description="What this task should accomplish"
    )

    depends_on: list[str] = Field(
        description="IDs of task that must finish first"
    )


class TaskExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        description="ID of the task that produced this result"
    )

    status: TaskExecutionStatus = Field(
        description="Outcome of the simulated task execution"
    )

    result: str | None = Field(
        default=None,
        description="Simulated output when the task completes",
    )

    error: str | None = Field(
        default=None,
        description="Failure or skip reason when execution does not complete",
    )


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description="short explanation of the plan"
    )

    missing_information: list[str] = Field(
        description="Information needed from the user"
    )

    tasks: list[PlannedTask] = Field(
        description="proposed tasks in the workflow"
    )
