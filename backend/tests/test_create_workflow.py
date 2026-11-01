
import pytest
from uuid import uuid4
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.models
from app.api.dependencies import get_current_user_id
import app.api.dependencies as api_dependencies
from app.api.dependencies import (
    get_workflow_plan_generator,
    get_workflow_task_runner,
)
from app.api.dependencies import require_workflow_executor
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.models.user import User
from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
    WorkflowPlan,
)


@pytest.fixture
def test_user_id():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(email="test@example.com")
        session.add(user)
        session.commit()
        session.refresh(user)
        user_id = user.id

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db_session] = (
        override_session
    )
    app.dependency_overrides[get_current_user_id] = (
        lambda: user_id
    )

    try:
        yield user_id
    finally:
        app.dependency_overrides.pop(
            get_db_session, None
        )
        app.dependency_overrides.pop(
            get_current_user_id, None
        )
        engine.dispose()


def test_create_workflow(test_user_id):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workflows/",
            json={
                "original_request": (
                    "Prepare a proposal for Client ABC"
                )
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["user_id"] == str(test_user_id)
        assert data["status"] == "PLANNING"

        list_response = client.get(
            "/api/v1/workflows/"
        )

        assert list_response.status_code == 200
        assert len(list_response.json()) == 1


def test_bearer_token_authenticates_configured_user(
    test_user_id,
    monkeypatch,
):
    settings = SimpleNamespace(
        workflow_api_token=SecretStr("local-test-api-token"),
        workflow_user_id=test_user_id,
    )
    monkeypatch.setattr(
        api_dependencies,
        "get_settings",
        lambda: settings,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/workflows/",
            headers={
                "Authorization": "Bearer local-test-api-token",
            },
        )

    assert response.status_code == 200
    assert response.json() == []


def test_reject_empty_request(test_user_id):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workflows/",
            json={"original_request": "   "},
        )

    assert response.status_code == 422


def test_get_workflow_by_id(test_user_id):
    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/workflows/",
            json={
                "original_request":
                    "Prepare proposal for Client ABC"
            },
        )

        assert create_response.status_code == 201
        workflow_id = create_response.json()["id"]

        response = client.get(
            f"/api/v1/workflows/{workflow_id}"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == workflow_id
    assert data["user_id"] == str(test_user_id)
    assert data["status"] == "PLANNING"
    assert data["original_request"] == (
        "Prepare proposal for Client ABC"
    )



def test_get_nonexistent_workflow(test_user_id):
    missing_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/workflows/{missing_id}"
        )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Workflow not found"
    )



def test_cannot_access_another_users_workflow(
    test_user_id,
):
    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/workflows/",
            json={
                "original_request":
                    "Confidential client proposal"
            },
        )

        assert create_response.status_code == 201
        workflow_id = create_response.json()["id"]

        # Simulate a different authenticated identity.
        other_user_id = uuid4()

        app.dependency_overrides[
            get_current_user_id
        ] = lambda: other_user_id

        response = client.get(
            f"/api/v1/workflows/{workflow_id}"
        )

    assert response.status_code == 404



def test_get_workflow_invalid_uuid(test_user_id):
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/workflows/not-a-valid-uuid"
        )

    assert response.status_code == 422



def test_status_update_requires_executor(test_user_id):
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/workflows/",
            json={
                "original_request": "Prepare client proposal"
            },
        )

        assert created.status_code == 201
        workflow_id = created.json()["id"]

        response = client.patch(
            f"/api/v1/workflows/{workflow_id}/status",
            json={
                "expected_status": "PLANNING",
                "new_status": "RUNNING",
            },
        )

    assert response.status_code == 403



def test_workflow_status_transitions(test_user_id):
    app.dependency_overrides[
        require_workflow_executor
    ] = lambda: None

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={
                    "original_request":
                        "Prepare client proposal"
                },
            )

            assert created.status_code == 201
            workflow_id = created.json()["id"]
            url = f"/api/v1/workflows/{workflow_id}/status"

            # Valid transition
            response = client.patch(
                url,
                json={
                    "expected_status": "PLANNING",
                    "new_status": "RUNNING",
                },
            )

            assert response.status_code == 200
            assert response.json()["status"] == "RUNNING"

            # Stale expected status
            stale = client.patch(
                url,
                json={
                    "expected_status": "PLANNING",
                    "new_status": "RUNNING",
                },
            )

            assert stale.status_code == 409

            # Invalid transition
            invalid = client.patch(
                url,
                json={
                    "expected_status": "RUNNING",
                    "new_status": "CANCELLED",
                },
            )

            assert invalid.status_code == 409

            # Verify persisted state
            saved = client.get(
                f"/api/v1/workflows/{workflow_id}"
            )

            assert saved.status_code == 200
            assert saved.json()["status"] == "RUNNING"

    finally:
        app.dependency_overrides.pop(
            require_workflow_executor, None
        )




def test_status_update_enforces_ownership(test_user_id):
    app.dependency_overrides[
        require_workflow_executor
    ] = lambda: None

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={
                    "original_request":
                        "Private client proposal"
                },
            )

            assert created.status_code == 201
            workflow_id = created.json()["id"]

            # Use an identity that does not own the workflow.
            other_user_id = uuid4()

            app.dependency_overrides[
                get_current_user_id
            ] = lambda: other_user_id

            response = client.patch(
                f"/api/v1/workflows/{workflow_id}/status",
                json={
                    "expected_status": "PLANNING",
                    "new_status": "RUNNING",
                },
            )

            assert response.status_code == 404

    finally:
        app.dependency_overrides.pop(
            require_workflow_executor, None
        )
        # The existing test_user_id fixture also
        # cleans up the user identity override.



def test_cancel_workflow(test_user_id):
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/workflows/",
            json={
                "original_request":
                    "Prepare proposal for Client ABC"
            },
        )

        assert created.status_code == 201

        workflow_id = created.json()["id"]
        url = f"/api/v1/workflows/{workflow_id}"

        cancelled = client.post(f"{url}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLED"

        # Repeating cancellation doesn't create
        # another workflow or change its status.
        repeated = client.post(f"{url}/cancel")

        assert repeated.status_code == 200
        assert repeated.json()["id"] == workflow_id

        # Cancellation preserves history.
        saved = client.get(url)

        assert saved.status_code == 200
        assert saved.json()["status"] == "CANCELLED"

        listed = client.get("/api/v1/workflows/")

        assert listed.status_code == 200
        assert len(listed.json()) == 1



def test_cancel_nonexistent_workflow(test_user_id):
    missing_id = uuid4()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/workflows/{missing_id}/cancel"
        )

    assert response.status_code == 404



def test_cannot_cancel_another_users_workflow(
    test_user_id,
):
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/workflows/",
            json={
                "original_request":
                    "Confidential client proposal"
            },
        )

        assert created.status_code == 201
        workflow_id = created.json()["id"]

        app.dependency_overrides[
            get_current_user_id
        ] = lambda: uuid4()

        response = client.post(
            f"/api/v1/workflows/{workflow_id}/cancel"
        )

    assert response.status_code == 404



def test_validation_error_format(test_user_id):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workflows/",
            json={"original_request": "   "},
        )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "VALIDATION_ERROR"
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0

    # The raw invalid request should not be echoed.
    assert "input" not in data["detail"][0]



def test_not_found_error_format(test_user_id):
    missing_workflow_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/workflows/{missing_workflow_id}"
        )

    assert response.status_code == 404

    assert response.json() == {
        "error": "NOT_FOUND",
        "detail": "Workflow not found",
    }



def test_unexpected_errors_are_sanitized(
    test_user_id,
):
    original_override = app.dependency_overrides[
        get_db_session
    ]

    def broken_session():
        raise RuntimeError(
            "PRIVATE_INTERNAL_DETAILS"
        )

    app.dependency_overrides[
        get_db_session
    ] = broken_session

    try:
        with TestClient(
            app,
            raise_server_exceptions=False,
        ) as client:
            response = client.get(
                "/api/v1/workflows/"
            )

        assert response.status_code == 500

        data = response.json()

        assert data["error"] == (
            "INTERNAL_SERVER_ERROR"
        )

        assert "PRIVATE_INTERNAL_DETAILS" not in (
            response.text
        )

    finally:
        app.dependency_overrides[
            get_db_session
        ] = original_override


def test_execute_workflow_returns_results_and_prevents_duplicate_runs(
    test_user_id,
    offline_task_runner,
):
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare client proposal",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find previous pricing",
                    depends_on=[],
                ),
            ],
        )

    app.dependency_overrides[require_workflow_executor] = lambda: None
    app.dependency_overrides[get_workflow_plan_generator] = (
        lambda: fake_planner
    )
    app.dependency_overrides[get_workflow_task_runner] = (
        lambda: offline_task_runner
    )

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={
                    "original_request": "Prepare a client proposal"
                },
            )
            workflow_id = created.json()["id"]

            response = client.post(
                f"/api/v1/workflows/{workflow_id}/execute"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "COMPLETED"
            assert data["planning_outcome"] == "READY"
            assert data["execution_results"][0]["status"] == "SIMULATED"

            duplicate = client.post(
                f"/api/v1/workflows/{workflow_id}/execute"
            )
            assert duplicate.status_code == 409

            saved = client.get(
                f"/api/v1/workflows/{workflow_id}"
            )
            assert saved.json()["status"] == "COMPLETED"
    finally:
        app.dependency_overrides.pop(
            require_workflow_executor,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_plan_generator,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_task_runner,
            None,
        )


def test_execute_workflow_waits_for_missing_information(
    test_user_id,
):
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Need client information",
            missing_information=["Which client is this for?"],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find previous pricing",
                    depends_on=[],
                ),
            ],
        )

    executed_ids = []

    def recording_runner(task, dependency_results):
        executed_ids.append(task.task_id)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    app.dependency_overrides[require_workflow_executor] = lambda: None
    app.dependency_overrides[get_workflow_plan_generator] = (
        lambda: fake_planner
    )
    app.dependency_overrides[get_workflow_task_runner] = (
        lambda: recording_runner
    )

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={"original_request": "Prepare a proposal"},
            )
            workflow_id = created.json()["id"]

            response = client.post(
                f"/api/v1/workflows/{workflow_id}/execute"
            )

        assert response.status_code == 200
        assert response.json()["status"] == "WAITING_FOR_INPUT"
        assert response.json()["execution_results"] == []
        assert executed_ids == []
    finally:
        app.dependency_overrides.pop(
            require_workflow_executor,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_plan_generator,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_task_runner,
            None,
        )


def test_execute_workflow_persists_partial_success(test_user_id):
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare proposal and check calendar",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find previous pricing",
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
            ],
        )

    def failing_runner(task, dependency_results):
        if task.task_id == "T1":
            raise RuntimeError("Pricing source unavailable")
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Test output for {task.task_id}",
        )

    app.dependency_overrides[require_workflow_executor] = lambda: None
    app.dependency_overrides[get_workflow_plan_generator] = (
        lambda: fake_planner
    )
    app.dependency_overrides[get_workflow_task_runner] = (
        lambda: failing_runner
    )

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={"original_request": "Prepare a proposal"},
            )
            workflow_id = created.json()["id"]

            response = client.post(
                f"/api/v1/workflows/{workflow_id}/execute"
            )

            assert response.status_code == 200
            assert response.json()["status"] == "PARTIAL_SUCCESS"

            saved = client.get(
                f"/api/v1/workflows/{workflow_id}"
            )
            assert saved.json()["status"] == "PARTIAL_SUCCESS"
    finally:
        app.dependency_overrides.pop(
            require_workflow_executor,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_plan_generator,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_task_runner,
            None,
        )


def test_execute_workflow_marks_failed_run(test_user_id):
    def failing_planner(request: str) -> WorkflowPlan:
        raise RuntimeError("Planner is temporarily unavailable")

    app.dependency_overrides[require_workflow_executor] = lambda: None
    app.dependency_overrides[get_workflow_plan_generator] = (
        lambda: failing_planner
    )

    try:
        with TestClient(
            app,
            raise_server_exceptions=False,
        ) as client:
            created = client.post(
                "/api/v1/workflows/",
                json={"original_request": "Prepare a proposal"},
            )
            workflow_id = created.json()["id"]

            response = client.post(
                f"/api/v1/workflows/{workflow_id}/execute"
            )

            assert response.status_code == 500
            assert response.json()["error"] == "INTERNAL_SERVER_ERROR"

            saved = client.get(
                f"/api/v1/workflows/{workflow_id}"
            )
            assert saved.json()["status"] == "FAILED"
    finally:
        app.dependency_overrides.pop(
            require_workflow_executor,
            None,
        )
        app.dependency_overrides.pop(
            get_workflow_plan_generator,
            None,
        )


def test_execute_workflow_requires_executor(test_user_id):
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/workflows/",
            json={"original_request": "Prepare a proposal"},
        )
        workflow_id = created.json()["id"]

        response = client.post(
            f"/api/v1/workflows/{workflow_id}/execute"
        )

    assert response.status_code == 403
