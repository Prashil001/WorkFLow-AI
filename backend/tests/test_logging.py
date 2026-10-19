
import json
import logging
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.core.logging import JsonFormatter


def test_health_has_request_id():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]
    UUID(request_id)


def test_request_ids_are_unique():
    with TestClient(app) as client:
        first = client.get("/health")
        second = client.get("/health")

    assert (
        first.headers["X-Request-ID"]
        != second.headers["X-Request-ID"]
    )


def test_json_formatter():
    record = logging.LogRecord(
        name="workflow_ai",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="request_finished",
        args=(),
        exc_info=None,
    )

    record.request_id = "test-123"
    record.status_code = 200

    formatted = JsonFormatter().format(record)
    data = json.loads(formatted)

    assert data["event"] == "request_finished"
    assert data["request_id"] == "test-123"
    assert data["status_code"] == 200
