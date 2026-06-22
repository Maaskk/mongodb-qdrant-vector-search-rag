from vector_rag.contracts import RetrievalResult
from vector_rag.mongodb_track.hybrid import rank_fusion_pipeline, reciprocal_rank_fusion


def hit(chunk_id: str, rank: int, mode: str) -> RetrievalResult:
    return RetrievalResult(
        backend="mongodb",
        query_id="q1",
        rank=rank,
        chunk_id=chunk_id,
        score=1.0 / rank,
        latency_ms=2.0,
        search_mode=mode,
        filter_name="none",
        run_id="run-1",
    )


def test_rrf_deduplicates_and_combines_rank_contributions() -> None:
    vector_hits = [hit("shared", 1, "ann"), hit("vector-only", 2, "ann")]
    text_hits = [hit("shared", 1, "text"), hit("text-only", 2, "text")]

    fused = reciprocal_rank_fusion(vector_hits, text_hits, rank_constant=60)

    assert [item.chunk_id for item in fused] == ["shared", "vector-only", "text-only"]
    assert fused[0].score == 2 / 61
    assert all(item.search_mode == "hybrid_rrf" for item in fused)


def test_rrf_respects_weights_and_top_k() -> None:
    vector_hits = [hit("vector", 1, "ann")]
    text_hits = [hit("text", 1, "text")]

    fused = reciprocal_rank_fusion(
        vector_hits, text_hits, vector_weight=2.0, text_weight=1.0, top_k=1
    )

    assert [item.chunk_id for item in fused] == ["vector"]


def test_rank_fusion_pipeline_contains_independent_vector_and_text_inputs(
    vector_384: list[float],
) -> None:
    pipeline = rank_fusion_pipeline(
        vector_384,
        "MongoDB RAG",
        top_k=5,
        num_candidates=100,
        vector_index="vector_index",
        text_index="text_index",
    )

    rank_fusion = pipeline[0]["$rankFusion"]
    assert set(rank_fusion["input"]["pipelines"]) == {"vector", "text"}
    vector_stage = rank_fusion["input"]["pipelines"]["vector"][0]["$vectorSearch"]
    assert vector_stage["numCandidates"] == 100
    assert pipeline[-1] == {"$limit": 5}
