import math

import pytest

from vector_rag.contracts import Chunk
from vector_rag.mongodb_track.schema import chunk_document, validate_embedding


def test_chunk_document_uses_chunk_id_for_idempotency(
    sample_chunk: Chunk, vector_384: list[float]
) -> None:
    document = chunk_document(sample_chunk, vector_384, run_id="run-1")

    assert document["_id"] == sample_chunk.chunk_id
    assert document["content_hash"] == sample_chunk.content_hash
    assert document["ingestion"]["run_id"] == "run-1"
    assert document["schema_version"] == 1


def test_chunk_document_rejects_wrong_vector_dimension(sample_chunk: Chunk) -> None:
    with pytest.raises(ValueError, match="expected 384"):
        chunk_document(sample_chunk, [0.1, 0.2], run_id="run-1")


def test_embedding_rejects_non_finite_values(vector_384: list[float]) -> None:
    vector_384[12] = math.nan

    with pytest.raises(ValueError, match="finite"):
        validate_embedding(vector_384, dimensions=384)

