"""Information-retrieval and latency metrics used by the benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {item for item, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranking[:k])
    return len(relevant.intersection(retrieved)) / len(relevant)


def mrr_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {item for item, grade in qrels.items() if grade > 0}
    for rank, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    def dcg(grades: Sequence[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    observed = [int(qrels.get(item, 0)) for item in ranking[:k]]
    ideal = sorted((int(grade) for grade in qrels.values()), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(observed) / ideal_score if ideal_score else 0.0


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
