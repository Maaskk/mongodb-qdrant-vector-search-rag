from collections.abc import Mapping, Sequence

from vector_rag.backend import VectorBackend
from vector_rag.contracts import Chunk, RetrievalResult


class FakeBackend:
    name = "fake"

    def ingest(self, chunks: Sequence[Chunk]) -> int:
        return len(chunks)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        query_id: str,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        return []

    def healthcheck(self) -> bool:
        return True


def test_backend_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeBackend(), VectorBackend)
