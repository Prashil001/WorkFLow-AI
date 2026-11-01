from app.workflow_engine.executor import execute_task
from app.workflow_engine.mock_data import load_pricing_documents_by_ids
from app.workflow_engine.rag import chunk_documents
from app.workflow_engine.schemas import PlannedTask, TaskExecutionResult


def test_chunk_documents_preserves_overlap_and_source_context():
    chunks = chunk_documents(
        [{
            "document_id": "doc-1",
            "client": "Example Co",
            "title": "Pricing Guide",
            "content": "one two three four five six seven eight",
        }],
        chunk_size=4,
        overlap=2,
    )

    assert len(chunks) == 3
    assert chunks[0].text.endswith("one two three four")
    assert chunks[1].text.endswith("three four five six")
    assert chunks[0].source_id == "doc-1"
    assert "Pricing Guide for Example Co" in chunks[0].text


def test_retriever_ranks_relevant_mock_document(offline_retriever):
    matches = offline_retriever.retrieve("Northstar Labs standard pricing")

    assert matches
    assert matches[0].source_id == "mock-pricing-001"
    assert matches[0].score > 0.9


def test_retriever_returns_no_results_for_unrelated_query(offline_retriever):
    assert offline_retriever.retrieve("quantum yacht") == []
    assert offline_retriever.retrieve("   ") == []


def test_retrieved_source_ids_feed_grounded_draft(offline_retriever):
    retrieval_task = PlannedTask(
        task_id="T1",
        task_type="retrieve_documents",
        description="Find Northstar Labs pricing",
        depends_on=[],
    )
    retrieval_result = execute_task(
        retrieval_task,
        retriever=offline_retriever,
    )
    captured = {}

    def fake_draft_generator(task, source_documents, dependency_results):
        captured["documents"] = source_documents
        captured["dependencies"] = dependency_results
        return "Implementation ranges from USD 12,000 to USD 18,000 [source:mock-pricing-001]."

    draft_task = PlannedTask(
        task_id="T2",
        task_type="draft_proposal",
        description="Draft a Northstar Labs proposal",
        depends_on=["T1"],
    )
    draft_result = execute_task(
        draft_task,
        {"T1": retrieval_result},
        retriever=offline_retriever,
        draft_generator=fake_draft_generator,
    )

    assert retrieval_result.source_ids == ["mock-pricing-001"]
    assert draft_result.status == "SIMULATED"
    assert draft_result.source_ids == ["mock-pricing-001"]
    assert "[source:mock-pricing-001]" in draft_result.result
    assert [doc["document_id"] for doc in captured["documents"]] == [
        "mock-pricing-001"
    ]
    assert captured["dependencies"] == {"T1": retrieval_result}


def test_source_loader_ignores_unknown_ids():
    documents = load_pricing_documents_by_ids(
        ["mock-pricing-001", "unknown-source"]
    )

    assert [doc["document_id"] for doc in documents] == [
        "mock-pricing-001"
    ]
