"""Safe runtime configuration for the MongoDB track."""

from __future__ import annotations

import os
from dataclasses import dataclass


class MongoConfigError(ValueError):
    """Raised when required MongoDB configuration is invalid or missing."""


@dataclass(frozen=True, repr=False)
class MongoConfig:
    """Connection and index settings, loaded without ever exposing credentials."""

    uri: str | None
    database: str = "vector_rag"
    collection: str = "chunks"
    vector_index: str = "vector_index"
    text_index: str = "text_index"
    dimensions: int = 384
    timeout_ms: int = 10_000

    @classmethod
    def from_env(cls, *, require_uri: bool = True) -> MongoConfig:
        uri = os.getenv("MONGODB_URI")
        if require_uri and not uri:
            raise MongoConfigError(
                "MONGODB_URI is required for live MongoDB operations. "
                "Use require_uri=False only for offline validation."
            )
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
        if dimensions <= 0:
            raise MongoConfigError("EMBEDDING_DIMENSIONS must be a positive integer")
        return cls(
            uri=uri,
            database=os.getenv("MONGODB_DATABASE", "vector_rag"),
            collection=os.getenv("MONGODB_COLLECTION", "chunks"),
            vector_index=os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
            text_index=os.getenv("MONGODB_TEXT_INDEX", "text_index"),
            dimensions=dimensions,
            timeout_ms=int(os.getenv("MONGODB_TIMEOUT_MS", "10000")),
        )

    def __repr__(self) -> str:
        uri = "<redacted>" if self.uri else "None"
        return (
            "MongoConfig("
            f"uri={uri}, database={self.database!r}, collection={self.collection!r}, "
            f"vector_index={self.vector_index!r}, text_index={self.text_index!r}, "
            f"dimensions={self.dimensions}, timeout_ms={self.timeout_ms})"
        )
