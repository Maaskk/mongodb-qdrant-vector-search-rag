"""Embedding adapters for live experiments and deterministic offline validation."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Any


class HashingEmbedder:
    """Dependency-free feature hashing; suitable for tests, never labeled as a live model."""

    model_name = "deterministic-feature-hashing-v1"
    _token = re.compile(r"[\w-]+", re.UNICODE)

    def __init__(self, *, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._token.findall(text.lower()):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have the same length")
        return sum(a * b for a, b in zip(left, right, strict=True))


class SentenceTransformerEmbedder:
    """Lazy adapter around the frozen live embedding model."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        module = importlib.import_module("sentence_transformers")
        model_class: Any = module.SentenceTransformer
        self._model: Any = model_class(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: Any = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(value) for value in vector] for vector in vectors]
