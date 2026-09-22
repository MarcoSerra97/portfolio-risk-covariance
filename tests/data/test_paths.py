"""Tests for the paths module."""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_risk.data import paths


class TestFindProjectRoot:
    def test_finds_pyproject_in_ancestor(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert paths._find_project_root(nested) == tmp_path.resolve()

    def test_finds_pyproject_in_start_dir(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        assert paths._find_project_root(tmp_path) == tmp_path.resolve()

    def test_accepts_file_argument(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        some_file = tmp_path / "some_module.py"
        some_file.write_text("")
        assert paths._find_project_root(some_file) == tmp_path.resolve()

    def test_raises_when_no_pyproject(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Could not find pyproject.toml"):
            paths._find_project_root(tmp_path)


class TestModuleConstants:
    def test_project_root_contains_pyproject(self) -> None:
        assert (paths.PROJECT_ROOT / "pyproject.toml").exists()

    def test_data_dir_under_project_root(self) -> None:
        assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"

    def test_raw_dir_under_data_dir(self) -> None:
        assert paths.RAW_DIR == paths.DATA_DIR / "raw"

    def test_processed_dir_under_data_dir(self) -> None:
        assert paths.PROCESSED_DIR == paths.DATA_DIR / "processed"

    def test_manifest_path_under_data_dir(self) -> None:
        assert paths.MANIFEST_PATH == paths.DATA_DIR / "manifest.yaml"


class TestEnsureDataDirs:
    def test_creates_missing_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = tmp_path / "data" / "raw"
        processed = tmp_path / "data" / "processed"
        monkeypatch.setattr(paths, "RAW_DIR", raw)
        monkeypatch.setattr(paths, "PROCESSED_DIR", processed)
        assert not raw.exists()
        assert not processed.exists()
        paths.ensure_data_dirs()
        assert raw.is_dir()
        assert processed.is_dir()

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = tmp_path / "data" / "raw"
        processed = tmp_path / "data" / "processed"
        raw.mkdir(parents=True)
        processed.mkdir(parents=True)
        monkeypatch.setattr(paths, "RAW_DIR", raw)
        monkeypatch.setattr(paths, "PROCESSED_DIR", processed)
        paths.ensure_data_dirs()  # should not raise
        assert raw.is_dir()
        assert processed.is_dir()
