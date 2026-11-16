import pytest

from app.evaluation.run_v2 import CONFIG_BUILDERS
from app.rag.hybrid import (
    HybridRetriever,
    RerankingRetriever,
    reciprocal_rank_fusion,
)
from app.workflow_engine.rag import RetrievedChunk


def _chunk(source_id, index=0, score=0.5):
    return RetrievedChunk(source_id, "Docs", "Title", index, f"{source_id}-{index}", score)


def _keys(chunks):
    return [(chunk.source_id, chunk.chunk_index) for chunk in chunks]


class ListRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.limits = []

    def retrieve(self, query, *, limit=3):
        self.limits.append(limit)
        return self.chunks[:limit]


def test_rrf_scores_match_hand_computed_values():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    lexical = [_chunk("c"), _chunk("a")]

    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    # a: 1/61 + 1/62, c: 1/63 + 1/61, b: 1/62
    assert _keys(fused) == [("a", 0), ("c", 0), ("b", 0)]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2].score == pytest.approx(1 / 62)


def test_rrf_treats_chunks_of_one_page_separately_and_breaks_ties_stably():
    fused = reciprocal_rank_fusion([[_chunk("b", 1), _chunk("a", 0)], [_chunk("a", 0), _chunk("b", 1)]])

    assert _keys(fused) == [("a", 0), ("b", 1)]  # equal scores -> source_id order


def test_hybrid_asks_both_retrievers_for_candidates_and_truncates():
    dense = ListRetriever([_chunk(f"d{i}") for i in range(10)])
    lexical = ListRetriever([_chunk(f"l{i}") for i in range(10)])

    results = HybridRetriever(dense, lexical, candidates=50).retrieve("q", limit=4)

    assert dense.limits == [50] and lexical.limits == [50]
    assert len(results) == 4
    assert HybridRetriever(dense, lexical).retrieve("   ") == []


class ReverseScorer:
    """Scores later candidates higher, so reranking reverses the head."""

    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append(list(documents))
        return [float(index) for index in range(len(documents))]


def test_reranker_reorders_head_and_keeps_tail_in_base_order():
    base = ListRetriever([_chunk(f"p{i}") for i in range(6)])
    scorer = ReverseScorer()

    results = RerankingRetriever(base, scorer, depth=3).retrieve("q", limit=6)

    assert base.limits == [6]
    assert len(scorer.calls[0]) == 3
    assert _keys(results) == [("p2", 0), ("p1", 0), ("p0", 0), ("p3", 0), ("p4", 0), ("p5", 0)]
    assert results[0].score == 2.0


def test_reranker_fetches_at_least_its_depth_and_validates_scores():
    base = ListRetriever([_chunk(f"p{i}") for i in range(30)])

    results = RerankingRetriever(base, ReverseScorer(), depth=20).retrieve("q", limit=3)

    assert base.limits == [20] and len(results) == 3

    class BrokenScorer:
        def score(self, query, documents):
            return [1.0]

    with pytest.raises(ValueError, match="unexpected number"):
        RerankingRetriever(base, BrokenScorer(), depth=5).retrieve("q")
    assert RerankingRetriever(base, ReverseScorer()).retrieve("") == []


def test_all_phase_four_configs_are_registered():
    assert set(CONFIG_BUILDERS) == {
        "vector", "hybrid", "hybrid-rerank", "hybrid-rerank-minilm",
    }
