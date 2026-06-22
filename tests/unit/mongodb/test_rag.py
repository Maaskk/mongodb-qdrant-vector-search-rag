from vector_rag.contracts import Chunk, EvaluationQuery
from vector_rag.mongodb_track.rag import (
    INSUFFICIENT_EVIDENCE,
    ContextAssembler,
    ExtractiveGenerator,
    MongoRAGPipeline,
)


class StaticGenerator:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, query: str, context: str) -> str:
        assert query
        assert "retrieved_context" in context
        return self.response


class TimeoutGenerator:
    def generate(self, query: str, context: str) -> str:
        del query, context
        raise TimeoutError("provider timed out")


def query() -> EvaluationQuery:
    return EvaluationQuery(
        query_id="q1",
        text="Where are embeddings stored?",
        expected_answer="In MongoDB documents.",
    )


def test_context_wraps_untrusted_chunks(sample_chunk: Chunk) -> None:
    context = ContextAssembler(max_characters=2000).assemble([sample_chunk])

    assert '<retrieved_context untrusted="true">' in context
    assert '<retrieved_chunk id="mongo:0">' in context
    assert sample_chunk.text in context


def test_context_respects_character_budget(sample_chunk: Chunk) -> None:
    context = ContextAssembler(max_characters=180).assemble([sample_chunk])

    assert len(context) <= 180
    assert context.endswith("</retrieved_context>")


def test_empty_evidence_returns_refusal() -> None:
    pipeline = MongoRAGPipeline(StaticGenerator("should not run"), run_id="run-1")

    result = pipeline.answer(query(), [])

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.cited_chunk_ids == []


def test_valid_citations_are_preserved(sample_chunk: Chunk) -> None:
    pipeline = MongoRAGPipeline(
        StaticGenerator("Embeddings live with metadata [mongo:0]."), run_id="run-1"
    )

    result = pipeline.answer(query(), [sample_chunk], retrieval_latency_ms=4.0)

    assert result.cited_chunk_ids == ["mongo:0"]
    assert result.answer.startswith("Embeddings")
    assert result.total_latency_ms >= result.retrieval_latency_ms


def test_unknown_or_missing_citations_produce_refusal(sample_chunk: Chunk) -> None:
    for response in ("Unsupported claim [missing:9].", "Unsupported claim."):
        pipeline = MongoRAGPipeline(StaticGenerator(response), run_id="run-1")

        result = pipeline.answer(query(), [sample_chunk])

        assert result.answer == INSUFFICIENT_EVIDENCE
        assert result.cited_chunk_ids == []


def test_generator_timeout_produces_safe_refusal(sample_chunk: Chunk) -> None:
    pipeline = MongoRAGPipeline(TimeoutGenerator(), run_id="run-1")

    result = pipeline.answer(query(), [sample_chunk])

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.cited_chunk_ids == []


def test_extractive_generator_is_credential_free_and_cites_chunk(sample_chunk: Chunk) -> None:
    context = ContextAssembler().assemble([sample_chunk])

    answer = ExtractiveGenerator().generate(query().text, context)

    assert sample_chunk.text in answer
    assert "[mongo:0]" in answer
