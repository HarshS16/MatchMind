"""IR evaluation metrics: Precision@K, Recall@K, NDCG@K, MRR.

Relevance labels are graded 0-3. Binary relevance for P/R/MRR = label >= 2.
"""

import math
from collections.abc import Sequence

BINARY_THRESHOLD = 2


def precision_at_k(ranked_labels: Sequence[int], k: int) -> float:
    if k <= 0 or not ranked_labels:
        return 0.0
    top = list(ranked_labels)[:k]
    relevant = sum(1 for rel in top if rel >= BINARY_THRESHOLD)
    return relevant / min(k, len(top)) if top else 0.0


def recall_at_k(ranked_labels: Sequence[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    top = list(ranked_labels)[:k]
    relevant = sum(1 for rel in top if rel >= BINARY_THRESHOLD)
    return relevant / total_relevant


def dcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """DCG with graded relevance: sum((2^rel - 1) / log2(i+1))."""
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(list(ranked_labels)[:k])
    )


def ndcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    ideal = sorted(ranked_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / idcg


def reciprocal_rank(ranked_labels: Sequence[int]) -> float:
    for i, rel in enumerate(ranked_labels):
        if rel >= BINARY_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def mrr(all_ranked_labels: Sequence[Sequence[int]]) -> float:
    if not all_ranked_labels:
        return 0.0
    return sum(reciprocal_rank(r) for r in all_ranked_labels) / len(all_ranked_labels)


def compute_metrics(
    ranked_labels: Sequence[int],
    total_relevant: int,
    ks: tuple[int, ...] = (5,),
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Compute all requested metrics for a single query."""
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(ranked_labels, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_labels, k, total_relevant)
    metrics[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked_labels, ndcg_k)
    metrics["mrr"] = reciprocal_rank(ranked_labels)
    return metrics


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
