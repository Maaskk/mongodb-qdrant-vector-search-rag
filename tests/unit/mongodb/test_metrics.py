import math

import pytest

from vector_rag.mongodb_track.metrics import (
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)


def test_recall_mrr_and_ndcg_known_ranking() -> None:
    qrels = {"a": 2, "b": 1}
    ranking = ["x", "a", "b"]
    expected_dcg = 3 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = 3 / math.log2(2) + 1 / math.log2(3)

    assert recall_at_k(ranking, qrels, 3) == 1.0
    assert mrr_at_k(ranking, qrels, 10) == 0.5
    assert ndcg_at_k(ranking, qrels, 3) == pytest.approx(expected_dcg / ideal_dcg)


def test_metrics_handle_empty_relevance() -> None:
    assert recall_at_k(["a"], {}, 10) == 0.0
    assert mrr_at_k(["a"], {}, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_percentile_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)
