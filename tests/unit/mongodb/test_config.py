import pytest

from vector_rag.mongodb_track.config import MongoConfig, MongoConfigError


def test_config_repr_redacts_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://user:secret@localhost")

    config = MongoConfig.from_env()

    assert "secret" not in repr(config)
    assert "<redacted>" in repr(config)


def test_config_requires_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)

    with pytest.raises(MongoConfigError, match="MONGODB_URI"):
        MongoConfig.from_env()


def test_config_can_be_built_without_uri_for_offline_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)

    config = MongoConfig.from_env(require_uri=False)

    assert config.uri is None
    assert config.dimensions == 384

