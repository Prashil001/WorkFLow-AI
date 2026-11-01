import pytest

from app.workflow_engine.mock_data import load_mock_business_data
from app.workflow_engine.rag import InMemoryRetriever
from app.workflow_engine.schemas import TaskExecutionResult


@pytest.fixture
def offline_task_runner():
    def run_task(task, dependency_results):
        source_ids = sorted({
            source_id
            for result in dependency_results.values()
            for source_id in result.source_ids
        })
        return TaskExecutionResult(
            task_id=task.task_id,
            status="SIMULATED",
            result=f"Offline test output for {task.task_id}",
            source_ids=source_ids,
        )

    return run_task


class KeywordEmbedder:
    """Small deterministic embedder for tests; no model files or network."""

    def _vector(self, text):
        normalized = text.lower()
        if any(word in normalized for word in ("northstar", "standard pricing", "implementation packages")):
            return [1.0, 0.0, 0.0]
        if any(word in normalized for word in ("cedar", "retail", "training")):
            return [0.0, 1.0, 0.0]
        if any(word in normalized for word in ("bluebird", "healthcare", "onboarding")):
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    def embed(self, documents):
        return [self._vector(document) for document in documents]

    def query_embed(self, query):
        return [self._vector(query)]


@pytest.fixture
def offline_retriever():
    data = load_mock_business_data()
    return InMemoryRetriever(
        data["pricing_documents"],
        KeywordEmbedder(),
        minimum_score=0.2,
    )


@pytest.fixture(autouse=True)
def use_offline_graph_runner(monkeypatch):
    """Keep graph tests deterministic and independent of external models."""
    import app.workflow_engine.graph as graph_module

    monkeypatch.setattr(graph_module, "execute_task", offline_default_task)


def offline_default_task(task, dependency_results):
    source_ids = sorted({
        source_id
        for result in dependency_results.values()
        for source_id in result.source_ids
    })
    return TaskExecutionResult(
        task_id=task.task_id,
        status="SIMULATED",
        result=f"Offline test output for {task.task_id}",
        source_ids=source_ids,
    )
