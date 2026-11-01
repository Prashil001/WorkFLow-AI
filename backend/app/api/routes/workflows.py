from typing import Annotated
from collections.abc import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_current_user_id,
    get_workflow_plan_generator,
    get_workflow_task_runner,
    require_workflow_executor,
)
from app.db.session import get_db_session
from app.schemas.workflow import (
    WorkflowExecutionRead,
    WorkflowRead,
    WorkFlowStatusUpdate,
    WorkflowStatus,
)
from app.models.workflow import Workflow
from app.schemas.workflow import WorkflowCreate
from app.workflow_engine.executor import TaskRunner
from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.schemas import WorkflowPlan
from app.services.workflow_service import (
    create_workflow,
    UserNotProvisionedError,
    list_workflows,
    get_workflow_by_id,
    update_workflow_status,
    WorkflowStatusConflict,
    cancel_workflow,
    WorkflowCancellationConflict,
)


def _final_workflow_status(
    planning_outcome: str,
    execution_results: list[dict[str, object]],
) -> WorkflowStatus:
    if planning_outcome == "NEEDS_INPUT":
        return "WAITING_FOR_INPUT"

    successful_count = sum(
        result.get("status") == "SIMULATED"
        for result in execution_results
    )

    if execution_results and successful_count == len(execution_results):
        return "COMPLETED"

    if successful_count:
        return "PARTIAL_SUCCESS"

    return "FAILED"

router = APIRouter(
    prefix="/api/v1/workflows",
    tags=["Workflows"],
)

@router.get("/",response_model=list[WorkflowRead])
def get_workflows(
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session: Annotated[Session,Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Workflow]:
    return list_workflows(
        session=session,
        user_id=user_id,
        limit=limit,
        offset=offset
    )


@router.post(
    "/",
    response_model=WorkflowRead,
    status_code=status.HTTP_201_CREATED,
)
def create_new_workflow(
    payload: WorkflowCreate,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
) -> Workflow:
    try:
        return create_workflow(
            session=session,
            user_id=user_id,
            original_request=payload.original_request
        )
    except UserNotProvisionedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not provisioned"
        )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowRead,
)
def get_single_workflow(
    workflow_id: UUID,
    user_id: Annotated[
        UUID, Depends(get_current_user_id)
    ],
    session: Annotated[
        Session, Depends(get_db_session)
    ],
) -> Workflow:
    workflow = get_workflow_by_id(
        session=session,
        workflow_id=workflow_id,
        user_id=user_id
    )
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow


@router.patch(
    "/{workflow_id}/status",
    response_model=WorkflowRead,
    dependencies=[Depends(require_workflow_executor)],
)
def change_workflow_status(
    workflow_id: UUID,
    payload: WorkFlowStatusUpdate,
    user_id: Annotated[
        UUID, Depends(get_current_user_id)
    ],
    session: Annotated[
        Session, Depends(get_db_session)
    ],
) -> Workflow:

    try:
        workflow = update_workflow_status(
            session=session,
            workflow_id=workflow_id,
            user_id=user_id,
            expected_status=payload.expected_status,
            new_status=payload.new_status,
        )
    except WorkflowStatusConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workflow not found",
        )

    return workflow


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowRead,
)
def cancel_existing_workflow(
    workflow_id: UUID,
    user_id: Annotated[
        UUID, Depends(get_current_user_id)
    ],
    session: Annotated[
        Session, Depends(get_db_session)
    ]
) -> Workflow:
    try:
        workflow = cancel_workflow(
            session=session,
            workflow_id=workflow_id,
            user_id=user_id
        )
    except WorkflowCancellationConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail = str(exc),
        ) from exc

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workflow not found",
        )

    return workflow


@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowExecutionRead,
    dependencies=[Depends(require_workflow_executor)],
)
def execute_existing_workflow(
    workflow_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
    plan_generator: Annotated[
        Callable[[str], WorkflowPlan],
        Depends(get_workflow_plan_generator),
    ],
    task_runner: Annotated[
        TaskRunner,
        Depends(get_workflow_task_runner),
    ],
) -> WorkflowExecutionRead:
    try:
        workflow = update_workflow_status(
            session=session,
            workflow_id=workflow_id,
            user_id=user_id,
            expected_status="PLANNING",
            new_status="RUNNING",
        )
    except WorkflowStatusConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    original_request = workflow.original_request
    session.rollback()

    graph = build_workflow_graph(
        plan_generator=plan_generator,
        task_runner=task_runner,
    )

    try:
        state = graph.invoke({
            "original_request": original_request,
        })
        plan = WorkflowPlan.model_validate(state["plan"])
        execution_results = state.get("execution_results", [])
        final_status = _final_workflow_status(
            state["planning_outcome"],
            execution_results,
        )
    except Exception:
        update_workflow_status(
            session=session,
            workflow_id=workflow_id,
            user_id=user_id,
            expected_status="RUNNING",
            new_status="FAILED",
        )
        raise

    try:
        update_workflow_status(
            session=session,
            workflow_id=workflow_id,
            user_id=user_id,
            expected_status="RUNNING",
            new_status=final_status,
        )
    except WorkflowStatusConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return WorkflowExecutionRead(
        workflow_id=workflow_id,
        status=final_status,
        planning_outcome=state["planning_outcome"],
        summary=plan.summary,
        missing_information=plan.missing_information,
        tasks=plan.tasks,
        execution_order=state.get("execution_order", []),
        execution_results=execution_results,
    )
