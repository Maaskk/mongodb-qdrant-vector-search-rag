"""Validated readers for versioned JSON Lines inputs."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    """Load JSON Lines records and report the exact invalid input line."""

    records: list[ModelT] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                records.append(model.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(f"{path}:{line_number}: invalid record: {error}") from error
    return records
