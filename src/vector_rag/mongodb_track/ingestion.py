"""Resilient and idempotent batched ingestion for MongoDB."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pymongo import UpdateOne
from pymongo.errors import AutoReconnect, NetworkTimeout, PyMongoError, ServerSelectionTimeoutError

from vector_rag.contracts import Chunk
from vector_rag.mongodb_track.schema import IngestionSummary, chunk_document

TRANSIENT_ERRORS = (AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError)


class MongoIngestor:
    """Bulk upsert chunks while skipping unchanged content and preserving failures."""

    def __init__(
        self,
        collection: Any,
        *,
        dimensions: int = 384,
        batch_size: int = 100,
        max_attempts: int = 3,
        dead_letter_dir: Path = Path("results/mongodb/dead_letters"),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size <= 0 or max_attempts <= 0:
            raise ValueError("batch_size and max_attempts must be positive")
        self.collection = collection
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.dead_letter_dir = dead_letter_dir
        self.sleep = sleep

    def ingest(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        run_id: str,
    ) -> IngestionSummary:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        started = time.perf_counter()
        upserted = updated = skipped = failed = 0
        dead_letter_path: Path | None = None

        for offset in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[offset : offset + self.batch_size]
            batch_embeddings = embeddings[offset : offset + self.batch_size]
            ids = [chunk.chunk_id for chunk in batch_chunks]
            existing = {
                str(item["_id"]): str(item.get("content_hash", ""))
                for item in self.collection.find(
                    {"_id": {"$in": ids}}, {"_id": 1, "content_hash": 1}
                )
            }
            documents: list[dict[str, Any]] = []
            operations: list[UpdateOne] = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=True):
                if existing.get(chunk.chunk_id) == chunk.content_hash:
                    skipped += 1
                    continue
                document = chunk_document(
                    chunk,
                    embedding,
                    run_id=run_id,
                    dimensions=self.dimensions,
                )
                documents.append(document)
                operations.append(
                    UpdateOne({"_id": chunk.chunk_id}, {"$set": document}, upsert=True)
                )
            if not operations:
                continue

            try:
                result = self._write_with_retry(operations)
                upserted += int(result.upserted_count)
                updated += int(result.modified_count)
            except PyMongoError as error:
                failed += len(documents)
                dead_letter_path = self._write_dead_letters(documents, run_id=run_id, error=error)

        return IngestionSummary(
            run_id=run_id,
            accepted=len(chunks),
            upserted=upserted,
            updated=updated,
            skipped=skipped,
            failed=failed,
            elapsed_seconds=time.perf_counter() - started,
            dead_letter_path=dead_letter_path,
        )

    def _write_with_retry(self, operations: list[UpdateOne]) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.collection.bulk_write(operations, ordered=False)
            except TRANSIENT_ERRORS:
                if attempt == self.max_attempts:
                    raise
                self.sleep(float(2 ** (attempt - 1)))
        raise AssertionError("unreachable retry state")

    def _write_dead_letters(
        self,
        documents: list[dict[str, Any]],
        *,
        run_id: str,
        error: PyMongoError,
    ) -> Path:
        self.dead_letter_dir.mkdir(parents=True, exist_ok=True)
        path = self.dead_letter_dir / f"{run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for document in documents:
                payload = {
                    "run_id": run_id,
                    "chunk_id": document["chunk_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "document": document,
                }
                handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        return path
