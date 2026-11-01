import json

from app.workflow_engine.executor import execute_task
from app.workflow_engine.mock_tools import (
    check_calendar_availability,
    search_pricing_documents,
)
from app.workflow_engine.schemas import PlannedTask


def test_search_pricing_documents_returns_ranked_fictional_matches():
    result = json.loads(
        search_pricing_documents("Northstar Labs implementation pricing")
    )

    assert result["source"] == "mock_local_dataset"
    assert result["matches"][0]["document_id"] == "mock-pricing-001"
    assert "fictional sample data" in result["notice"]


def test_search_pricing_documents_returns_empty_for_no_match():
    result = json.loads(
        search_pricing_documents("unlisted client quantum yacht")
    )

    assert result["matches"] == []


def test_calendar_tool_filters_by_date_and_time_of_day():
    result = json.loads(
        check_calendar_availability("2026-11-04 afternoon")
    )

    assert [slot["slot_id"] for slot in result["slots"]] == [
        "mock-slot-002"
    ]
    assert "not a live calendar" in result["notice"]


def test_executor_dispatches_read_only_task_types_to_tools():
    task = PlannedTask(
        task_id="T1",
        task_type="retrieve_documents",
        description="Search Northstar Labs pricing",
        depends_on=[],
    )

    result = execute_task(task)

    assert result.status == "SIMULATED"
    assert "mock-pricing-001" in result.result


