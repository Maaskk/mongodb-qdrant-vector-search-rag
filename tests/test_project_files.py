from pathlib import Path


def test_required_project_files_exist() -> None:
    for path in (
        "pyproject.toml",
        ".env.example",
        "docker-compose.yml",
        "configs/embeddings.yaml",
        "configs/benchmark.yaml",
    ):
        assert Path(path).is_file(), path
