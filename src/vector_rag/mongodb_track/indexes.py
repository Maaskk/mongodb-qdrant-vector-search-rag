"""Create and monitor Atlas Search and Vector Search indexes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pymongo.operations import SearchIndexModel

from vector_rag.mongodb_track.schema import IndexStatus


def load_index_definition(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        definition = cast(dict[str, Any], json.load(handle))
    required = {"name", "type", "definition"}
    missing = required.difference(definition)
    if missing:
        raise ValueError(f"index definition {path} is missing: {sorted(missing)}")
    return definition


class IndexManager:
    """Idempotently manages the two search indexes required by this project."""

    def __init__(
        self,
        collection: Any,
        definitions: list[dict[str, Any]],
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.collection = collection
        self.definitions = definitions
        self.sleep = sleep

    @classmethod
    def from_files(
        cls,
        collection: Any,
        vector_path: Path,
        text_path: Path,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> IndexManager:
        return cls(
            collection,
            [load_index_definition(vector_path), load_index_definition(text_path)],
            sleep=sleep,
        )

    @property
    def target_names(self) -> set[str]:
        return {str(definition["name"]) for definition in self.definitions}

    def ensure_indexes(self) -> list[str]:
        existing = {str(item["name"]) for item in self.collection.list_search_indexes()}
        pending = [item for item in self.definitions if item["name"] not in existing]
        if not pending:
            return []
        models: list[SearchIndexModel] = []
        for item in pending:
            kwargs: dict[str, Any] = {
                "definition": item["definition"],
                "name": item["name"],
            }
            if item["type"] != "search":
                kwargs["type"] = item["type"]
            models.append(SearchIndexModel(**kwargs))
        self.collection.create_search_indexes(models=models)
        return [str(item["name"]) for item in pending]

    def statuses(self) -> list[IndexStatus]:
        statuses: list[IndexStatus] = []
        for item in self.collection.list_search_indexes():
            name = str(item.get("name", ""))
            if name not in self.target_names:
                continue
            status = str(item.get("status", "UNKNOWN"))
            queryable = bool(item.get("queryable", status.upper() == "READY"))
            statuses.append(IndexStatus(name=name, status=status, queryable=queryable))
        return statuses

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 600,
        poll_interval_seconds: float = 5,
    ) -> list[IndexStatus]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            statuses = self.statuses()
            ready = {item.name for item in statuses if item.queryable}
            if ready == self.target_names:
                return statuses
            if time.monotonic() >= deadline:
                missing = sorted(self.target_names.difference(ready))
                raise TimeoutError(f"search indexes did not become ready: {missing}")
            self.sleep(poll_interval_seconds)
