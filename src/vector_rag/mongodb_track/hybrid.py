"""Client-side and MongoDB-native reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vector_rag.contracts import RetrievalResult
from vector_rag.mongodb_track.schema import validate_embedding


@dataclass
class _FusionEntry:
    template: RetrievalResult
    score: float = 0.0


def reciprocal_rank_fusion(
    vector_hits: Sequence[RetrievalResult],
    text_hits: Sequence[RetrievalResult],
    *,
    rank_constant: int = 60,
    vector_weight: float = 1.0,
    text_weight: float = 1.0,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse rankings deterministically while keeping the shared result contract."""

    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    entries: dict[str, _FusionEntry] = {}
    for hits, weight in ((vector_hits, vector_weight), (text_hits, text_weight)):
        for hit in hits:
            entry = entries.setdefault(hit.chunk_id, _FusionEntry(template=hit))
            entry.score += weight / (rank_constant + hit.rank)
    ordered = sorted(entries.values(), key=lambda entry: entry.score, reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        RetrievalResult(
            backend="mongodb",
            query_id=entry.template.query_id,
            rank=rank,
            chunk_id=entry.template.chunk_id,
            score=entry.score,
            latency_ms=entry.template.latency_ms,
            search_mode="hybrid_rrf",
            filter_name=entry.template.filter_name,
            run_id=entry.template.run_id,
        )
        for rank, entry in enumerate(ordered, start=1)
    ]


def rank_fusion_pipeline(
    query_vector: Sequence[float],
    query_text: str,
    *,
    top_k: int,
    num_candidates: int,
    vector_index: str,
    text_index: str,
    dimensions: int = 384,
    vector_weight: float = 1.0,
    text_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Build MongoDB 8.0+'s native ``$rankFusion`` hybrid pipeline."""

    vector = validate_embedding(query_vector, dimensions=dimensions)
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": vector_index,
                                    "path": "embedding",
                                    "queryVector": vector,
                                    "numCandidates": num_candidates,
                                    "limit": top_k,
                                }
                            }
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": text_index,
                                    "text": {
                                        "query": query_text,
                                        "path": ["title", "text"],
                                    },
                                }
                            }
                        ],
                    }
                },
                "combination": {"weights": {"vector": vector_weight, "text": text_weight}},
                "scoreDetails": True,
            }
        },
        {"$limit": top_k},
    ]
