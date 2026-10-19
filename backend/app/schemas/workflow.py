from datetime import datetime
from uuid import UUID
from typing import Annotated, Literal

from pydantic import StringConstraints
from pydantic import BaseModel, ConfigDict


class WorkflowRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    graph_thread_id: str
    original_request: str
    status: str
    created_at: datetime


class WorkflowCreate(BaseModel):
    original_request: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=5000,
        )
    ]

WorkflowStatus = Literal[
    "PLANNING",
    "RUNNING",
    "WAITING_FOR_INPUT",
    "WAITING_FOR_APPROVAL",
    "COMPLETED",
    "PARTIAL_SUCCESS",
    "FAILED",
    "CANCELLED"
]

class WorkFlowStatusUpdate(BaseModel):
    expected_status: WorkflowStatus
    new_status: WorkflowStatus

