"""Stable backend boundary implemented independently by each contributor."""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from vector_rag.contracts import Chunk, RetrievalResult


@runtime_checkable
class VectorBackend(Protocol):
    """Minimum behavior required by the shared benchmark runner."""

    name: str

    def ingest(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks and return the number accepted."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        query_id: str,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked results in the shared artifact format."""

    def healthcheck(self) -> bool:
        """Return whether the backend is ready to accept requests."""

