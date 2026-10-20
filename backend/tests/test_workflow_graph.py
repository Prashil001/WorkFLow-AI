
import pytest

from app.workflow_engine.graph import build_workflow_graph


def test_graph_accepts_valid_request():
    graph = build_workflow_graph()

    result = graph.invoke({
        "original_request": "  Prepare client proposal  "
    })

    assert result["normalized_request"] == (
        "Prepare client proposal"
    )

    assert result["is_valid"] is True


def test_graph_preserves_original_request():
    graph = build_workflow_graph()

    original = "  Find the previous proposal  "

    result = graph.invoke({
        "original_request": original
    })

    assert result["original_request"] == original


@pytest.mark.parametrize(
    "invalid_request",
    [
        "",
        "   ",
        "x" * 5001,
    ],
)
def test_graph_rejects_invalid_requests(
    invalid_request,
):
    graph = build_workflow_graph()

    with pytest.raises(ValueError):
        graph.invoke({
            "original_request": invalid_request
        })


def test_graph_accepts_maximum_length():
    graph = build_workflow_graph()

    result = graph.invoke({
        "original_request": "x" * 5000
    })

    assert result["is_valid"] is True
