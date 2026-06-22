from pathlib import Path
from typing import Any

from pymongo.errors import AutoReconnect

from vector_rag.contracts import Chunk
from vector_rag.mongodb_track.ingestion import MongoIngestor


class FakeBulkResult:
    def __init__(self, *, upserted: int, matched: int) -> None:
        self.upserted_count = upserted
        self.matched_count = matched
        self.modified_count = matched


class FakeCollection:
    def __init__(
        self,
        existing: dict[str, str] | None = None,
        failures: list[Exception] | None = None,
    ) -> None:
        self.existing = dict(existing or {})
        self.failures = list(failures or [])
        self.calls: list[list[Any]] = []

    def find(self, query: dict[str, Any], projection: dict[str, int]):
        del projection
        ids = query["_id"]["$in"]
        return [
            {"_id": chunk_id, "content_hash": self.existing[chunk_id]}
            for chunk_id in ids
            if chunk_id in self.existing
        ]

    def bulk_write(self, operations: list[Any], *, ordered: bool) -> FakeBulkResult:
        assert ordered is False
        self.calls.append(operations)
        if self.failures:
            raise self.failures.pop(0)
        upserted = 0
        matched = 0
        for operation in operations:
            chunk_id = operation._filter["_id"]
            if chunk_id in self.existing:
                matched += 1
            else:
                upserted += 1
            self.existing[chunk_id] = operation._doc["$set"]["content_hash"]
        return FakeBulkResult(upserted=upserted, matched=matched)


def test_ingestion_upserts_changed_chunks_and_skips_unchanged(
    sample_chunk: Chunk, vector_384: list[float], tmp_path: Path
) -> None:
    collection = FakeCollection(existing={sample_chunk.chunk_id: sample_chunk.content_hash})
    changed = sample_chunk.model_copy(
        update={"chunk_id": "mongo:1", "chunk_index": 1, "content_hash": "b" * 64}
    )
    ingestor = MongoIngestor(collection, dead_letter_dir=tmp_path, sleep=lambda _: None)

    summary = ingestor.ingest([sample_chunk, changed], [vector_384, vector_384], run_id="run-1")

    assert summary.accepted == 2
    assert summary.skipped == 1
    assert summary.upserted == 1
    assert summary.updated == 0
    assert summary.failed == 0
    assert len(collection.calls) == 1


def test_ingestion_retries_transient_mongodb_errors(
    sample_chunk: Chunk, vector_384: list[float], tmp_path: Path
) -> None:
    collection = FakeCollection(failures=[AutoReconnect("temporary")])
    sleeps: list[float] = []
    ingestor = MongoIngestor(
        collection,
        dead_letter_dir=tmp_path,
        max_attempts=2,
        sleep=sleeps.append,
    )

    summary = ingestor.ingest([sample_chunk], [vector_384], run_id="run-retry")

    assert summary.upserted == 1
    assert summary.failed == 0
    assert sleeps == [1.0]
    assert len(collection.calls) == 2


def test_ingestion_dead_letters_exhausted_batches(
    sample_chunk: Chunk, vector_384: list[float], tmp_path: Path
) -> None:
    collection = FakeCollection(failures=[AutoReconnect("one"), AutoReconnect("two")])
    ingestor = MongoIngestor(
        collection,
        dead_letter_dir=tmp_path,
        max_attempts=2,
        sleep=lambda _: None,
    )

    summary = ingestor.ingest([sample_chunk], [vector_384], run_id="run-failed")

    assert summary.failed == 1
    assert summary.dead_letter_path == tmp_path / "run-failed.jsonl"
    assert summary.dead_letter_path.exists()
    assert '"chunk_id": "mongo:0"' in summary.dead_letter_path.read_text()


def test_ingestion_rejects_chunk_embedding_count_mismatch(
    sample_chunk: Chunk, tmp_path: Path
) -> None:
    ingestor = MongoIngestor(FakeCollection(), dead_letter_dir=tmp_path)

    try:
        ingestor.ingest([sample_chunk], [], run_id="bad")
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected a ValueError")
