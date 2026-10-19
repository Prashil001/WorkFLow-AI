from uuid import UUID

from sqlalchemy import select, func, update
from sqlalchemy.orm import Session

from app.models.workflow import Workflow
from app.models.user import User

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "PLANNING": {"RUNNING"},
    "RUNNING": {
        "WAITING_FOR_INPUT",
        "WAITING_FOR_APPROVAL",
        "FAILED",
    },
    "WAITING_FOR_INPUT": {
        "RUNNING",
        "FAILED",
    },
    "WAITING_FOR_APPROVAL": {
        "RUNNING",
        "FAILED",
    },
    "COMPLETED": set(),
    "PARTIAL_SUCCESS": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

class UserNotProvisionedError(Exception):
    pass

class WorkflowStatusConflict(Exception):
    pass

class WorkflowCancellationConflict(Exception):
    pass

def list_workflows(
        session: Session,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
) -> list[Workflow]:
    statement = (
        select(Workflow)
        .where(Workflow.user_id == user_id)
        .order_by(
            Workflow.created_at.desc(),
            Workflow.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    return list(session.scalars(statement).all())


def create_workflow(
        session: Session,
        user_id: UUID,
        original_request: str,
) -> Workflow:
    with session.begin():
        user = session.get(User, user_id)

        if user is None:
            raise UserNotProvisionedError("Authenticated user is not provisioned")

        workflow = Workflow(
            user_id = user_id,
            original_request = original_request,
            status = "PLANNING",
        )
        session.add(workflow)
        session.flush()
        session.refresh(workflow)

    return workflow


def get_workflow_by_id(
        session: Session,
        workflow_id: UUID,
        user_id: UUID,
) -> Workflow | None:

    statement = select(Workflow).where(
        Workflow.id == workflow_id,
        Workflow.user_id == user_id, 
    )
    return session.scalar(statement)


def update_workflow_status(
    session: Session,
    workflow_id: UUID,
    user_id: UUID,
    expected_status: str,
    new_status: str,
) -> Workflow | None:

    with session.begin():
        workflow = session.scalar(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.user_id == user_id,
            )
        )

        if workflow is None:
            return None

        if workflow.status != expected_status:
            raise WorkflowStatusConflict(
                "workflow status has changed"
            )

        allowed = ALLOWED_TRANSITIONS.get(
            workflow.status, set()
        )

        if new_status not in allowed:
            raise WorkflowStatusConflict(
                "Invalid workflow status transition"
            )

        statement = (
            update(Workflow.__table__)
            .where(
                Workflow.__table__.c.id == workflow_id,
                Workflow.__table__.c.user_id == user_id,
                Workflow.__table__.c.status == expected_status
            )
            .values(
                status = new_status,
                updated_at = func.now(),
            )
        )
        result = session.execute(statement=statement)

        if result.rowcount != 1:
            raise WorkflowStatusConflict(
                "Concurrent workflow modification detected"
            )

        session.refresh(workflow)

    return workflow


def cancel_workflow(
        session: Session,
        workflow_id: UUID,
        user_id: UUID,
) -> Workflow | None:

    with session.begin():
        workflow = session.scalar(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.user_id == user_id,
            )
        )

        if workflow is None:
            return workflow

        if workflow.status == "CANCELLED":
            return workflow

        if workflow.status != "PLANNING":
            raise WorkflowCancellationConflict(
                "Only planning workflows can be cancelled"
            )

        statement = (
            update(Workflow.__table__)
            .where(
                Workflow.__table__.c.id == workflow_id,
                Workflow.__table__.c.user_id == user_id,
                workflow.__table__.c.status == "PLANNING",
            )
            .values(
                status = "CANCELLED",
                updated_at = func.now(),
            )
        )
        result = session.execute(statement=statement)
        if result.rowcount != 1:
            raise WorkflowCancellationConflict(
                "workflow changed during cancellation"
            )

        session.refresh(workflow)

    return workflow        

