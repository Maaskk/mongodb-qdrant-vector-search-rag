from collections.abc import Iterator

import pytest

from vector_rag.contracts import Chunk


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="mongo:0",
        document_id="mongo",
        title="MongoDB Vector Search",
        text="MongoDB stores embeddings with document metadata.",
        source="fixture",
        category="mongodb",
        language="en",
        chunk_index=0,
        content_hash="a" * 64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture
def vector_384() -> list[float]:
    return [0.01] * 384


class FakeIndexCollection:
    def __init__(self, indexes: list[dict[str, object]] | None = None) -> None:
        self.indexes = list(indexes or [])
        self.created_models: list[object] = []

    def list_search_indexes(self) -> Iterator[dict[str, object]]:
        return iter(self.indexes)

    def create_search_indexes(self, models: list[object]) -> list[str]:
        self.created_models.extend(models)
        return [model.document["name"] for model in models]  # type: ignore[attr-defined]


@pytest.fixture
def fake_index_collection() -> FakeIndexCollection:
    return FakeIndexCollection()
