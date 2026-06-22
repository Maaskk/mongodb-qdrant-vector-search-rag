"""MongoDB exact, approximate, filtered, and lexical retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

from vector_rag.contracts import RetrievalResult
from vector_rag.mongodb_track.schema import validate_embedding


class MongoRetriever:
    """Build and execute MongoDB Search aggregation pipelines."""

    def __init__(
        self,
        collection: Any,
        *,
        vector_index: str = "vector_index",
        text_index: str = "text_index",
        dimensions: int = 384,
        run_id: str = "interactive",
    ) -> None:
        self.collection = collection
        self.vector_index = vector_index
        self.text_index = text_index
        self.dimensions = dimensions
        self.run_id = run_id

    def vector_pipeline(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        exact: bool = False,
        num_candidates: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = validate_embedding(query_vector, dimensions=self.dimensions)
        vector_search: dict[str, Any] = {
            "index": self.vector_index,
            "path": "embedding",
            "queryVector": vector,
            "limit": top_k,
            "exact": exact,
        }
        if not exact:
            if num_candidates < top_k:
                raise ValueError("num_candidates must be greater than or equal to top_k")
            vector_search["numCandidates"] = num_candidates
        if filters:
            vector_search["filter"] = dict(filters)
        return [
            {"$vectorSearch": vector_search},
            {
                "$project": {
                    "_id": 1,
                    "chunk_id": 1,
                    "title": 1,
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

    def text_pipeline(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return [
            {
                "$search": {
                    "index": self.text_index,
                    "text": {"query": query, "path": ["title", "text"]},
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "chunk_id": 1,
                    "title": 1,
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
            {"$limit": top_k},
        ]

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        query_id: str,
        top_k: int,
        exact: bool = False,
        num_candidates: int = 100,
        filters: Mapping[str, Any] | None = None,
        filter_name: str = "none",
    ) -> list[RetrievalResult]:
        pipeline = self.vector_pipeline(
            query_vector,
            top_k=top_k,
            exact=exact,
            num_candidates=num_candidates,
            filters=filters,
        )
        started = time.perf_counter()
        documents = list(self.collection.aggregate(pipeline))
        latency_ms = (time.perf_counter() - started) * 1000
        mode = "exact" if exact else ("filtered_ann" if filters else "ann")
        return self._map_results(
            documents,
            query_id=query_id,
            latency_ms=latency_ms,
            search_mode=mode,
            filter_name=filter_name,
        )

    def search_text(
        self,
        query: str,
        *,
        query_id: str,
        top_k: int,
        filter_name: str = "none",
    ) -> list[RetrievalResult]:
        started = time.perf_counter()
        documents = list(self.collection.aggregate(self.text_pipeline(query, top_k=top_k)))
        latency_ms = (time.perf_counter() - started) * 1000
        return self._map_results(
            documents,
            query_id=query_id,
            latency_ms=latency_ms,
            search_mode="text",
            filter_name=filter_name,
        )

    def _map_results(
        self,
        documents: list[dict[str, Any]],
        *,
        query_id: str,
        latency_ms: float,
        search_mode: str,
        filter_name: str,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                backend="mongodb",
                query_id=query_id,
                rank=rank,
                chunk_id=str(document.get("chunk_id", document["_id"])),
                score=float(document.get("score", 0.0)),
                latency_ms=latency_ms,
                search_mode=search_mode,
                filter_name=filter_name,
                run_id=self.run_id,
            )
            for rank, document in enumerate(documents, start=1)
        ]
