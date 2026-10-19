
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user_id
from app.main import app


def test_requires_authentication():
    with TestClient(app) as client:
        response = client.get("/api/v1/workflows/")

    assert response.status_code == 401


def test_list_workflows_for_test_user():
    test_user_id = uuid4()

    app.dependency_overrides[get_current_user_id] = (
        lambda: test_user_id
    )

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workflows/")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.pop(
            get_current_user_id, None
        )
