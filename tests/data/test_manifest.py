"""Tests for the manifest module."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from portfolio_risk.data import manifest as mf


class TestComputeFileSha256:
    def test_known_value(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.txt"
        path.write_bytes(b"hello world")
        # Well-known SHA-256 of "hello world"
        assert mf.compute_file_sha256(path) == (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        assert mf.compute_file_sha256(path) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_deterministic(self, tmp_path: Path) -> None:
        path = tmp_path / "data.bin"
        path.write_bytes(b"the quick brown fox")
        h1 = mf.compute_file_sha256(path)
        h2 = mf.compute_file_sha256(path)
        assert h1 == h2

    def test_streaming_matches_whole(self, tmp_path: Path) -> None:
        """Chunked hashing must match whole-file hashing."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"x" * (65536 * 3 + 42))
        whole = mf.compute_file_sha256(path, chunk_size=10**9)
        chunked = mf.compute_file_sha256(path, chunk_size=1024)
        assert whole == chunked


class TestBuildManifest:
    def test_universe_counts(self, tmp_path: Path) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
        )
        assert manifest.n_instruments == 75
        assert manifest.bucket_counts == {
            "equity": 38,
            "srf_etf": 15,
            "rate_etf": 8,
            "commodity_fx": 8,
            "vol_spread": 6,
        }
        assert manifest.universe_version == mf.UNIVERSE_VERSION

    def test_window_recorded(self) -> None:
        start = date(2010, 1, 4)
        end = date(2024, 12, 31)
        manifest = mf.build_manifest(window_start=start, window_end=end, artefacts={})
        assert manifest.window_start == start
        assert manifest.window_end == end

    def test_default_fetched_at_is_recent(self) -> None:
        before = datetime.now(UTC)
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4), window_end=date(2024, 12, 31), artefacts={}
        )
        after = datetime.now(UTC)
        assert before <= manifest.fetched_at <= after

    def test_custom_fetched_at(self) -> None:
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
            fetched_at=ts,
        )
        assert manifest.fetched_at == ts

    def test_artefact_hashes(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        f1.write_bytes(b"content-a")
        f2.write_bytes(b"content-b")
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={"raw_prices": f1, "returns": f2},
        )
        assert "raw_prices_sha256" in manifest.integrity
        assert "returns_sha256" in manifest.integrity
        assert manifest.integrity["raw_prices_sha256"] == mf.compute_file_sha256(f1)
        assert manifest.integrity["returns_sha256"] == mf.compute_file_sha256(f2)

    def test_missing_artefacts_skipped(self, tmp_path: Path) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={"nonexistent": tmp_path / "does_not_exist.parquet"},
        )
        assert manifest.integrity == {}


class TestManifestToDict:
    def test_shape(self) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
            fetched_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        d = manifest.to_dict()
        assert set(d.keys()) == {"universe", "window", "fetched_at", "integrity"}
        assert d["universe"]["n_instruments"] == 75
        assert d["window"]["start"] == date(2010, 1, 4)


class TestWriteAndRead:
    def test_roundtrip(self, tmp_path: Path) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
            fetched_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC),
        )
        path = tmp_path / "manifest.yaml"
        mf.write_manifest(manifest, path)
        loaded = mf.read_manifest(path)
        assert loaded["universe"]["n_instruments"] == 75
        assert loaded["universe"]["buckets"]["equity"] == 38
        assert loaded["window"]["start"] == date(2010, 1, 4)

    def test_write_creates_parent_dir(self, tmp_path: Path) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4), window_end=date(2024, 12, 31), artefacts={}
        )
        path = tmp_path / "nested" / "dir" / "manifest.yaml"
        mf.write_manifest(manifest, path)
        assert path.exists()

    def test_written_yaml_is_human_readable(self, tmp_path: Path) -> None:
        artefact = tmp_path / "sample.parquet"
        artefact.write_bytes(b"hello world")
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={"raw_prices": artefact},
            fetched_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        path = tmp_path / "manifest.yaml"
        mf.write_manifest(manifest, path)
        text = path.read_text()
        # Block style, one key per line
        assert "universe:" in text
        assert "n_instruments: 75" in text
        # No flow-style braces (PyYAML uses {} only for empty dicts; with a
        # populated integrity block, block style is used throughout).
        assert "{" not in text

    def test_read_rejects_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(TypeError, match="not a mapping"):
            mf.read_manifest(path)

    def test_key_order_preserved_in_output(self, tmp_path: Path) -> None:
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
            fetched_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        path = tmp_path / "manifest.yaml"
        mf.write_manifest(manifest, path)
        text = path.read_text()
        # 'universe' should appear before 'window' before 'fetched_at'
        assert text.index("universe") < text.index("window") < text.index("fetched_at")


class TestYamlSemantics:
    def test_date_serialised_natively(self, tmp_path: Path) -> None:
        """PyYAML writes dates as YYYY-MM-DD, not as strings."""
        manifest = mf.build_manifest(
            window_start=date(2010, 1, 4),
            window_end=date(2024, 12, 31),
            artefacts={},
            fetched_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        path = tmp_path / "manifest.yaml"
        mf.write_manifest(manifest, path)
        with path.open() as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded["window"]["start"], date)
