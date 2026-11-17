import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.evaluation.metrics import (  # noqa: E402
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    # labels in ranked order; relevant = label >= 2
    ranked = [3, 0, 2, 1, 3]
    assert precision_at_k(ranked, 5) == 3 / 5
    assert precision_at_k(ranked, 3) == 2 / 3
    assert precision_at_k([], 5) == 0.0


def test_recall_at_k():
    ranked = [3, 0, 2, 1, 3]
    # total relevant (>=2) across corpus = 3
    assert recall_at_k(ranked, 5, 3) == 1.0
    assert recall_at_k(ranked, 1, 3) == 1 / 3
    assert recall_at_k(ranked, 5, 0) == 0.0


def test_ndcg_at_k_perfect():
    labels = [3, 2, 1, 0]
    assert ndcg_at_k(labels, 4) == 1.0


def test_ndcg_at_k_known_value():
    ranked = [3, 2, 3, 0, 1]
    ideal = sorted(ranked, reverse=True)
    expected = dcg_at_k(ranked, 10) / dcg_at_k(ideal, 10)
    assert math.isclose(ndcg_at_k(ranked, 10), expected)
    assert 0.0 < ndcg_at_k(ranked, 10) < 1.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank([0, 0, 2, 3]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert mrr([[2, 0], [0, 3], [0, 0]]) == (1.0 + 0.5 + 0.0) / 3


def test_binary_threshold():
    # label 1 is NOT relevant for P/R/MRR; label 2 is
    assert precision_at_k([1, 1, 2], 3) == 1 / 3
    assert reciprocal_rank([1, 2]) == 0.5
