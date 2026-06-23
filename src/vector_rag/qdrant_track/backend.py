"""Qdrant backend with lazy optional dependencies and deterministic artifacts."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

from vector_rag.backend import VectorBackend
from vector_rag.contracts import Chunk, RetrievalResult


class QdrantDependencyError(RuntimeError):
    """Raised when a live Qdrant dependency is requested but not installed."""


class Embedder(Protocol):
    """Minimal embedding interface used by the Qdrant track."""

    model_name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""


@dataclass(frozen=True)
class QdrantSettings:
    """Runtime settings for the Qdrant collection."""

    collection_name: str = "rag_corpus"
    url: str | None = "http://localhost:6333"
    storage_path: str | None = None
    vector_size: int = 384
    distance: str = "Cosine"
    run_id: str = "qdrant-run-001"


class SentenceTransformerEmbedder:
    """Lazy adapter around the frozen embedding model used by the project."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        module = _import_optional_module(
            "sentence_transformers",
            extra="embeddings",
            purpose="creating live Qdrant embeddings",
        )
        model_class: Any = module.SentenceTransformer
        self._model: Any = model_class(self.model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: Any = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(value) for value in vector] for vector in vectors]


def stable_point_id(chunk_id: str) -> int:
    """Return a deterministic positive integer Qdrant point id for a chunk id."""

    digest = sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class QdrantVectorBackend(VectorBackend):
    """Qdrant implementation of the shared VectorBackend protocol.

    The class can run against real Qdrant when the optional `qdrant` and
    `embeddings` extras are installed. Unit tests inject lightweight fakes so
    ordinary CI does not download models or require Docker.
    """

    name = "qdrant"

    def __init__(
        self,
        *,
        collection_name: str = "rag_corpus",
        url: str | None = "http://localhost:6333",
        storage_path: str | None = None,
        vector_size: int = 384,
        run_id: str = "qdrant-run-001",
        client: Any | None = None,
        embedder: Embedder | None = None,
        recreate_collection: bool = True,
    ) -> None:
        self.settings = QdrantSettings(
            collection_name=collection_name,
            url=url,
            storage_path=storage_path,
            vector_size=vector_size,
            run_id=run_id,
        )
        self.collection_name = self.settings.collection_name
        self.embedder = embedder or SentenceTransformerEmbedder()
        if self.embedder.dimensions != self.settings.vector_size:
            raise ValueError(
                f"embedder dimensions {self.embedder.dimensions} do not match "
                f"collection vector size {self.settings.vector_size}"
            )

        self._models = None
        if client is None:
            qdrant_client = _import_optional_module(
                "qdrant_client",
                extra="qdrant",
                purpose="connecting to Qdrant",
            )
            self._models = _import_optional_module(
                "qdrant_client.models",
                extra="qdrant",
                purpose="building Qdrant requests",
            )
            client_class: Any = qdrant_client.QdrantClient
            if self.settings.storage_path is not None:
                self.client: Any = client_class(path=self.settings.storage_path)
            else:
                self.client = client_class(url=self.settings.url)
        else:
            self.client = client

        self._init_collection(recreate_collection=recreate_collection)

    def _init_collection(self, *, recreate_collection: bool) -> None:
        if recreate_collection:
            with suppress(Exception):
                self.client.delete_collection(self.collection_name)

        if recreate_collection or not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._vector_params(),
            )

        for field_name in ("category", "language", "source"):
            create_index = getattr(self.client, "create_payload_index", None)
            if create_index is not None:
                with suppress(Exception):
                    create_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema="keyword",
                    )

    def _collection_exists(self) -> bool:
        collection_exists = getattr(self.client, "collection_exists", None)
        if collection_exists is not None:
            return bool(collection_exists(self.collection_name))
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _vector_params(self) -> Any:
        if self._models is None:
            return {"size": self.settings.vector_size, "distance": self.settings.distance}
        return self._models.VectorParams(
            size=self.settings.vector_size,
            distance=self._models.Distance.COSINE,
        )

    def healthcheck(self) -> bool:
        """Check if Qdrant is ready."""

        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def ingest(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks in Qdrant and return the number accepted."""

        if not chunks:
            return 0

        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        points: list[Any] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.settings.vector_size:
                raise ValueError(
                    f"chunk {chunk.chunk_id} produced vector length {len(vector)}, "
                    f"expected {self.settings.vector_size}"
                )
            points.append(self._point_for(chunk=chunk, vector=vector))

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def _point_for(self, *, chunk: Chunk, vector: Sequence[float]) -> Any:
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "text": chunk.text,
            "source": chunk.source,
            "category": chunk.category,
            "language": chunk.language,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "embedding_model": chunk.embedding_model,
        }
        if chunk.published_at is not None:
            payload["published_at"] = chunk.published_at.isoformat()

        point_id = stable_point_id(chunk.chunk_id)
        vector_values = [float(value) for value in vector]
        if self._models is None:
            return {"id": point_id, "vector": vector_values, "payload": payload}
        return self._models.PointStruct(id=point_id, vector=vector_values, payload=payload)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        query_id: str,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search and return ranked RetrievalResult objects."""

        start_time = time.perf_counter()
        query_filter = self._query_filter(filters)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": [float(value) for value in query_vector],
            "limit": top_k,
            "with_payload": True,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        response = self.client.query_points(**kwargs)
        hits = getattr(response, "points", response)
        latency_ms = (time.perf_counter() - start_time) * 1000
        filter_name = _filter_name(filters)

        return [
            RetrievalResult(
                backend=self.name,
                query_id=query_id,
                rank=rank,
                chunk_id=_payload_value(hit, "chunk_id", default="unknown"),
                score=float(_hit_value(hit, "score", default=0.0)),
                latency_ms=latency_ms,
                search_mode="dense",
                filter_name=filter_name,
                run_id=self.settings.run_id,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    def _query_filter(self, filters: Mapping[str, str] | None) -> Any | None:
        if not filters:
            return None
        ordered = [(key, filters[key]) for key in sorted(filters)]
        if self._models is None:
            return {
                "must": [
                    {"key": key, "match": {"value": value}}
                    for key, value in ordered
                ]
            }
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key=key,
                    match=self._models.MatchValue(value=value),
                )
                for key, value in ordered
            ]
        )


def _filter_name(filters: Mapping[str, str] | None) -> str:
    if not filters:
        return "none"
    return ";".join(f"{key}={filters[key]}" for key in sorted(filters))


def _hit_value(hit: Any, key: str, *, default: Any) -> Any:
    if isinstance(hit, Mapping):
        return hit.get(key, default)
    return getattr(hit, key, default)


def _payload_value(hit: Any, key: str, *, default: str) -> str:
    payload = _hit_value(hit, "payload", default={})
    if isinstance(payload, Mapping):
        value = payload.get(key, default)
        return str(value)
    value = getattr(payload, key, default)
    return str(value)


def _import_optional_module(module_name: str, *, extra: str, purpose: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise QdrantDependencyError(
            f"Install the `{extra}` extra before {purpose}: "
            f"python -m pip install -e '.[{extra}]'"
        ) from exc
