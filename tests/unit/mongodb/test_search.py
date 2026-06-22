from typing import Any

from vector_rag.mongodb_track.search import MongoRetriever


class FakeSearchCollection:
    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []
        self.pipelines: list[list[dict[str, Any]]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipelines.append(pipeline)
        return iter(self.documents)


def test_exact_pipeline_uses_enn(vector_384: list[float]) -> None:
    retriever = MongoRetriever(FakeSearchCollection())

    pipeline = retriever.vector_pipeline(vector_384, top_k=5, exact=True)

    stage = pipeline[0]["$vectorSearch"]
    assert stage["exact"] is True
    assert "numCandidates" not in stage


def test_ann_pipeline_uses_candidates_and_filters(vector_384: list[float]) -> None:
    retriever = MongoRetriever(FakeSearchCollection())

    pipeline = retriever.vector_pipeline(
        vector_384,
        top_k=5,
        exact=False,
        num_candidates=100,
        filters={"category": "mongodb", "language": "en"},
    )

    stage = pipeline[0]["$vectorSearch"]
    assert stage["exact"] is False
    assert stage["numCandidates"] == 100
    assert stage["filter"] == {"category": "mongodb", "language": "en"}


def test_text_pipeline_queries_title_and_text() -> None:
    retriever = MongoRetriever(FakeSearchCollection())

    pipeline = retriever.text_pipeline("vector databases", top_k=3)

    search = pipeline[0]["$search"]
    assert search["index"] == "text_index"
    assert search["text"]["path"] == ["title", "text"]
    assert pipeline[-1] == {"$limit": 3}


def test_vector_search_maps_documents_to_contract(vector_384: list[float]) -> None:
    collection = FakeSearchCollection(
        [{"_id": "mongo:0", "score": 0.91}, {"_id": "mongo:1", "score": 0.77}]
    )
    retriever = MongoRetriever(collection, run_id="run-search")

    results = retriever.search_vector(
        vector_384,
        query_id="q1",
        top_k=2,
        exact=False,
        num_candidates=50,
        filter_name="none",
    )

    assert [result.chunk_id for result in results] == ["mongo:0", "mongo:1"]
    assert [result.rank for result in results] == [1, 2]
    assert all(result.backend == "mongodb" for result in results)
    assert all(result.search_mode == "ann" for result in results)
    assert all(result.latency_ms >= 0 for result in results)


def test_exact_search_mode_is_named_exact(vector_384: list[float]) -> None:
    retriever = MongoRetriever(FakeSearchCollection([{"_id": "mongo:0", "score": 1.0}]))

    result = retriever.search_vector(
        vector_384, query_id="q1", top_k=1, exact=True
    )[0]

    assert result.search_mode == "exact"
