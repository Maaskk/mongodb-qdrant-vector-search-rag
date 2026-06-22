import json
from pathlib import Path

from vector_rag.mongodb_track.indexes import IndexManager, load_index_definition

from .conftest import FakeIndexCollection


def test_index_definitions_are_valid() -> None:
    vector = load_index_definition(Path("configs/mongodb/vector-index.json"))
    text = load_index_definition(Path("configs/mongodb/text-index.json"))

    assert vector["name"] == "vector_index"
    assert vector["type"] == "vectorSearch"
    assert text["name"] == "text_index"
    json.dumps(vector)
    json.dumps(text)


def test_index_manager_creates_only_missing_indexes(
    fake_index_collection: FakeIndexCollection,
) -> None:
    manager = IndexManager.from_files(
        fake_index_collection,
        Path("configs/mongodb/vector-index.json"),
        Path("configs/mongodb/text-index.json"),
    )

    created = manager.ensure_indexes()

    assert created == ["vector_index", "text_index"]
    assert len(fake_index_collection.created_models) == 2


def test_index_manager_skips_existing_index() -> None:
    collection = FakeIndexCollection([{"name": "vector_index", "status": "READY"}])
    manager = IndexManager.from_files(
        collection,
        Path("configs/mongodb/vector-index.json"),
        Path("configs/mongodb/text-index.json"),
    )

    assert manager.ensure_indexes() == ["text_index"]


def test_index_manager_waits_until_both_indexes_are_ready() -> None:
    class SequencedCollection(FakeIndexCollection):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def list_search_indexes(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            status = "BUILDING" if self.calls == 1 else "READY"
            return iter(
                [
                    {"name": "vector_index", "status": status, "queryable": status == "READY"},
                    {"name": "text_index", "status": status, "queryable": status == "READY"},
                ]
            )

    manager = IndexManager.from_files(
        SequencedCollection(),
        Path("configs/mongodb/vector-index.json"),
        Path("configs/mongodb/text-index.json"),
        sleep=lambda _: None,
    )

    statuses = manager.wait_until_ready(timeout_seconds=1, poll_interval_seconds=0)

    assert {status.name for status in statuses} == {"vector_index", "text_index"}
    assert all(status.ready for status in statuses)
