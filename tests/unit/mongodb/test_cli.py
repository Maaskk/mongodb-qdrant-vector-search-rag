from vector_rag.mongodb_track.__main__ import build_parser, main


def test_all_commands_parse_without_external_connections() -> None:
    parser = build_parser()

    assert parser.parse_args(["indexes"]).command == "indexes"
    assert parser.parse_args(["ingest"]).command == "ingest"
    assert parser.parse_args(["search", "vector metadata"]).command == "search"
    assert parser.parse_args(["rag", "Where are vectors stored?"]).command == "rag"
    assert parser.parse_args(["benchmark", "--offline-validation"]).command == "benchmark"


def test_live_command_without_uri_fails_concisely(monkeypatch, capsys) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)

    status = main(["indexes"])

    assert status == 2
    error = capsys.readouterr().err
    assert "MONGODB_URI is required" in error
    assert "Traceback" not in error
