
import pytest
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.models
from app.api.dependencies import get_current_user_id
from app.api.dependencies import require_workflow_executor
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.models.user import User


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
                    "new_status": "COMPLETED",
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
