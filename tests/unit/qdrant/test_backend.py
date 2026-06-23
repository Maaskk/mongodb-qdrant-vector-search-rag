from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vector_rag.contracts import Chunk
from vector_rag.qdrant_track.backend import QdrantSettings, QdrantVectorBackend, stable_point_id


@dataclass
class FakeHit:
    payload: dict[str, str]
    score: float


@dataclass
class FakeQueryResponse:
    points: list[FakeHit]


class FakeClient:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.indexes: list[tuple[str, str]] = []
        self.upserts: list[list[dict[str, object]]] = []
        self.last_query_filter: object | None = None

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collection = collection_name

    def create_collection(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: str
    ) -> None:
        self.indexes.append((field_name, field_schema))

    def get_collection(self, collection_name: str) -> dict[str, str]:
        return {"name": collection_name}

    def upsert(self, *, collection_name: str, points: list[dict[str, object]]) -> None:
        self.upserts.append(points)

    def query_points(self, **kwargs: object) -> FakeQueryResponse:
        self.last_query_filter = kwargs.get("query_filter")
        return FakeQueryResponse(
            points=[
                FakeHit(
                    payload={
                        "chunk_id": "chunk-a",
                        "document_id": "doc-a",
                        "text": "MongoDB vector",
                    },
                    score=0.91,
                )
            ]
        )


class FakeEmbedder:
    model_name = "fake-embedder"
    dimensions = 4

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def make_chunk(chunk_id: str = "chunk-a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-a",
        title="A",
        text="MongoDB vector search",
        source="fixture",
        category="database",
        language="en",
        chunk_index=0,
        content_hash="a" * 64,
        embedding_model="fake-embedder",
    )


def test_qdrant_settings_default_to_docker_service_url() -> None:
    assert QdrantSettings().url == "http://localhost:6333"
    assert QdrantSettings().storage_path is None


def test_qdrant_backend_imports_without_optional_dependencies() -> None:
    backend = QdrantVectorBackend(client=FakeClient(), embedder=FakeEmbedder(), vector_size=4)

    assert backend.healthcheck() is True


def test_stable_point_id_is_deterministic_and_positive() -> None:
    first = stable_point_id("chunk-a")
    second = stable_point_id("chunk-a")
    other = stable_point_id("chunk-b")

    assert first == second
    assert first != other
    assert first > 0


def test_ingest_upserts_payload_and_creates_keyword_indexes() -> None:
    client = FakeClient()
    backend = QdrantVectorBackend(client=client, embedder=FakeEmbedder(), vector_size=4)

    accepted = backend.ingest([make_chunk()])

    assert accepted == 1
    assert ("category", "keyword") in client.indexes
    assert ("language", "keyword") in client.indexes
    assert ("source", "keyword") in client.indexes
    point = client.upserts[0][0]
    assert point["id"] == stable_point_id("chunk-a")
    assert point["vector"] == [1.0, 0.0, 0.0, 0.0]
    assert point["payload"]["chunk_id"] == "chunk-a"  # type: ignore[index]


def test_search_forwards_filters_and_returns_shared_contract() -> None:
    client = FakeClient()
    backend = QdrantVectorBackend(client=client, embedder=FakeEmbedder(), vector_size=4)

    results = backend.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        query_id="q1",
        top_k=3,
        filters={"category": "database", "language": "en"},
    )

    assert client.last_query_filter == {
        "must": [
            {"key": "category", "match": {"value": "database"}},
            {"key": "language", "match": {"value": "en"}},
        ]
    }
    assert results[0].backend == "qdrant"
    assert results[0].query_id == "q1"
    assert results[0].chunk_id == "chunk-a"
    assert results[0].search_mode == "dense"
    assert results[0].filter_name == "category=database;language=en"
