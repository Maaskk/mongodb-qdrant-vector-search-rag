"""Validated MongoDB documents and operational result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

from vector_rag.contracts import Chunk


def validate_embedding(vector: Iterable[float], *, dimensions: int = 384) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != dimensions:
        raise ValueError(f"embedding has {len(values)} dimensions; expected {dimensions}")
    if not all(isfinite(value) for value in values):
        raise ValueError("embedding values must all be finite")
    return values


def chunk_document(
    chunk: Chunk,
    embedding: Iterable[float],
    *,
    run_id: str,
    dimensions: int = 384,
    indexed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the canonical idempotent MongoDB document for one chunk."""

    vector = validate_embedding(embedding, dimensions=dimensions)
    timestamp = indexed_at or datetime.now(UTC)
    return {
        "_id": chunk.chunk_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "text": chunk.text,
        "source": chunk.source,
        "category": chunk.category,
        "language": chunk.language,
        "published_at": chunk.published_at,
        "embedding": vector,
        "chunk_index": chunk.chunk_index,
        "embedding_model": chunk.embedding_model,
        "content_hash": chunk.content_hash,
        "schema_version": 1,
        "ingestion": {"run_id": run_id, "indexed_at": timestamp},
    }


@dataclass(frozen=True)
class IndexStatus:
    name: str
    status: str
    queryable: bool

    @property
    def ready(self) -> bool:
        return self.queryable and self.status.upper() == "READY"


@dataclass(frozen=True)
class IngestionSummary:
    run_id: str
    accepted: int = 0
    upserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    dead_letter_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.dead_letter_path is not None:
            result["dead_letter_path"] = str(self.dead_letter_path)
        return result
