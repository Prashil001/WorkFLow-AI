
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

import app.workflow_engine.planner as planner_module
import app.workflow_engine.drafting as drafting_module

from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.planner import (
    PlannerServiceError,
    generate_plan,
)
from app.workflow_engine.schemas import WorkflowPlan
from app.workflow_engine.schemas import PlannedTask
from pydantic import SecretStr


def example_plan() -> WorkflowPlan:
    return WorkflowPlan(
        summary="Prepare a client proposal",
        missing_information=[],
        tasks=[],
    )


def test_planner_receives_clean_request():
    fake_planner = Mock()
    fake_planner.invoke.return_value = example_plan()

    generate_plan(
        "   Prepare a client proposal   ",
        planner=fake_planner,
    )

    fake_planner.invoke.assert_called_once()

    messages = fake_planner.invoke.call_args.args[0]

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    assert messages[1].content == (
        "Prepare a client proposal"
    )


@pytest.mark.parametrize(
    "invalid_request",
    [
        "",
        "   ",
        "x" * 5001,
        None,
    ],
)
def test_invalid_input_never_calls_model(
    invalid_request,
):
    fake_planner = Mock()

    with pytest.raises(ValueError):
        generate_plan(
            invalid_request,
            planner=fake_planner,
        )

    fake_planner.invoke.assert_not_called()


def test_maximum_request_length():
    fake_planner = Mock()
    fake_planner.invoke.return_value = example_plan()

    generate_plan(
        "x" * 5000,
        planner=fake_planner,
    )

    messages = fake_planner.invoke.call_args.args[0]

    assert len(messages[1].content) == 5000


def test_missing_information_is_preserved():
    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare a client proposal",
            missing_information=[
                "Which client is this for?",
                "What services should be included?",
            ],
            tasks=[],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request": "Prepare a proposal"
    })

    assert result["plan"]["missing_information"] == [
        "Which client is this for?",
        "What services should be included?",
    ]

    assert result["plan"]["tasks"] == []


def test_invalid_provider_response_stops_graph():
    fake_provider = Mock()
    fake_provider.invoke.return_value = {
        "unexpected": "invalid result"
    }

    def injected_planner(request: str) -> WorkflowPlan:
        return generate_plan(
            request,
            planner=fake_provider,
        )

    graph = build_workflow_graph(
        plan_generator=injected_planner
    )

    with pytest.raises(
        PlannerServiceError,
        match="invalid response",
    ):
        graph.invoke({
            "original_request": "Prepare a proposal"
        })


def test_provider_timeout_is_handled():
    fake_provider = Mock()
    fake_provider.invoke.side_effect = TimeoutError(
        "SENSITIVE_PROVIDER_DETAILS"
    )

    with pytest.raises(PlannerServiceError) as error:
        generate_plan(
            "Prepare a proposal",
            planner=fake_provider,
        )

    assert str(error.value) == (
        "AI planner request failed"
    )

    assert "SENSITIVE_PROVIDER_DETAILS" not in (
        str(error.value)
    )

    assert isinstance(
        error.value.__cause__,
        TimeoutError,
    )


def test_missing_api_key(monkeypatch):
    fake_settings = SimpleNamespace(
        groq_api_key=None
    )

    monkeypatch.setattr(
        planner_module,
        "get_settings",
        lambda: fake_settings,
    )

    with pytest.raises(
        PlannerServiceError,
        match="GROQ_API_KEY is not configured",
    ):
        planner_module.create_planner()


def test_planner_uses_a_bounded_retry_and_timeout(monkeypatch):
    settings = SimpleNamespace(
        groq_api_key=SecretStr("test-key"),
        groq_model="test-model",
        llm_timeout_seconds=12,
        llm_max_retries=1,
    )
    monkeypatch.setattr(planner_module, "get_settings", lambda: settings)
    fake_model = Mock()
    fake_model.with_structured_output.return_value = "configured-planner"
    constructor = Mock(return_value=fake_model)
    monkeypatch.setattr(planner_module, "ChatGroq", constructor)

    assert planner_module.create_planner() == "configured-planner"
    constructor.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        temperature=0,
        timeout=12,
        max_retries=1,
    )


def test_drafting_model_uses_the_same_retry_and_timeout_limits(monkeypatch):
    settings = SimpleNamespace(
        groq_api_key=SecretStr("test-key"),
        groq_model="test-model",
        llm_timeout_seconds=9,
        llm_max_retries=1,
    )
    monkeypatch.setattr(drafting_module, "get_settings", lambda: settings)
    fake_model = Mock()
    fake_model.invoke.return_value = SimpleNamespace(
        content="A grounded draft [source:mock-pricing-001]."
    )
    constructor = Mock(return_value=fake_model)
    monkeypatch.setattr(drafting_module, "ChatGroq", constructor)

    output = drafting_module.generate_grounded_draft(
        PlannedTask(
            task_id="T1",
            task_type="draft_proposal",
            description="Draft a proposal",
            depends_on=[],
        ),
        [{
            "document_id": "mock-pricing-001",
            "title": "Pricing Guide",
            "client": "Northstar Labs",
            "content": "Implementation packages range from USD 12,000 to USD 18,000.",
        }],
        {},
    )

    assert "[source:mock-pricing-001]" in output
    constructor.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        temperature=0,
        timeout=9,
        max_retries=1,
    )
