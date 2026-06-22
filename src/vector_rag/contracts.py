"""Versioned data models shared by both independent backend tracks."""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    """Strict, immutable base model for reproducible artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Chunk(ContractModel):
    """One independently indexable section of a source document."""

    chunk_id: NonEmptyStr
    document_id: NonEmptyStr
    title: NonEmptyStr
    text: NonEmptyStr
    source: NonEmptyStr
    category: NonEmptyStr
    language: NonEmptyStr
    published_at: datetime | None = None
    chunk_index: int = Field(ge=0)
    content_hash: Sha256
    embedding_model: NonEmptyStr


class EvaluationQuery(ContractModel):
    """A benchmark query with an answer used for grounding checks."""

    query_id: NonEmptyStr
    text: NonEmptyStr
    expected_answer: NonEmptyStr
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(ContractModel):
    """One ranked item returned by a backend search."""

    backend: NonEmptyStr
    query_id: NonEmptyStr
    rank: int = Field(gt=0)
    chunk_id: NonEmptyStr
    score: float
    latency_ms: float = Field(ge=0)
    search_mode: NonEmptyStr
    filter_name: NonEmptyStr
    run_id: NonEmptyStr


class RAGResult(ContractModel):
    """A generated answer plus the chunk identifiers supporting it."""

    backend: NonEmptyStr
    query_id: NonEmptyStr
    answer: NonEmptyStr
    cited_chunk_ids: list[NonEmptyStr]
    retrieval_latency_ms: float = Field(ge=0)
    generation_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    run_id: NonEmptyStr

