import math

from vector_rag.mongodb_track.embeddings import HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=64)

    first = embedder.embed(["MongoDB vector search"])[0]
    second = embedder.embed(["MongoDB vector search"])[0]

    assert first == second
    assert len(first) == 64
    assert math.sqrt(sum(value * value for value in first)) == 1.0


def test_hashing_embedder_preserves_lexical_similarity() -> None:
    embedder = HashingEmbedder(dimensions=128)
    query, related, unrelated = embedder.embed(
        ["mongodb vector metadata", "mongodb stores vector metadata", "cooking pasta recipe"]
    )

    assert embedder.similarity(query, related) > embedder.similarity(query, unrelated)
