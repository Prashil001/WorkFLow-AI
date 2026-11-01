from app.evaluation.run import (
    evaluate_dataset,
    evaluate_grounding_cases,
    load_dataset,
)
from app.workflow_engine.rag import RetrievedChunk


class FixtureRetriever:
    def retrieve(self, query, *, limit=3):
        if "Northstar" in query:
            source_id = "mock-pricing-001"
        elif "Cedar" in query:
            source_id = "mock-pricing-002"
        elif "Bluebird" in query:
            source_id = "mock-pricing-003"
        else:
            return []

        return [RetrievedChunk(
            source_id=source_id,
            client="Fixture client",
            title="Fixture title",
            chunk_index=0,
            text="Fixture content",
            score=0.9,
        )][:limit]


def test_versioned_evaluation_dataset_passes_with_fixture_retriever():
    report = evaluate_dataset(retriever=FixtureRetriever())

    assert report["dataset"]["version"] == 1
    assert report["summary"] == {
        "passed_cases": 11,
        "total_cases": 11,
        "pass_rate": 1.0,
        "all_passed": True,
    }
    assert report["suites"]["retrieval"]["metrics"]["hit_at_1"] == 1.0
    assert report["suites"]["retrieval"]["metrics"]["precision_at_3"] == 1.0
    assert report["suites"]["retrieval"]["metrics"]["correct_empty_rate"] == 1.0


def test_evaluation_dataset_has_a_stable_version_and_hash():
    dataset, dataset_hash = load_dataset()

    assert dataset["dataset_name"] == "workflow-ai-regression"
    assert dataset["version"] == 1
    assert len(dataset_hash) == 64


def test_grounding_evaluator_rejects_unknown_source_citations():
    results = evaluate_grounding_cases([{
        "id": "unknown-source",
        "draft": "Pricing is USD 100 [source:unknown-doc].",
        "allowed_source_ids": ["mock-pricing-001"],
        "required_source_ids": ["mock-pricing-001"],
        "expected_integrity": False,
    }])

    assert results["all_passed"] is True
    assert results["cases"][0]["observed_integrity"] is False
