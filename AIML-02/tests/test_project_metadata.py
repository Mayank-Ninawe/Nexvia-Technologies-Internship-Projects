from pathlib import Path


def test_project_metadata_file_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "README.md").is_file()