"""Reproducibility manifest for the on-disk data cache.

Writes ``data/manifest.yaml`` per the design document §6: a small
human-readable file recording the universe version, the fetch window,
the fetch timestamp, and integrity hashes for the parquet artefacts.

The manifest is tracked in git even though the parquet files themselves
are not — anyone cloning the repo can rerun ``acquisition.py`` and
verify that their locally-fetched cache matches ours by comparing
hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from portfolio_risk.data import universe

UNIVERSE_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class Manifest:
    """Structured manifest, serialisable to YAML."""

    universe_version: int
    n_instruments: int
    bucket_counts: dict[str, int]
    window_start: date
    window_end: date
    fetched_at: datetime
    integrity: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a YAML-serialisable dict."""
        return {
            "universe": {
                "version": self.universe_version,
                "n_instruments": self.n_instruments,
                "buckets": dict(self.bucket_counts),
            },
            "window": {
                "start": self.window_start,
                "end": self.window_end,
            },
            "fetched_at": self.fetched_at,
            "integrity": dict(self.integrity),
        }


def compute_file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(
    window_start: date,
    window_end: date,
    artefacts: dict[str, Path],
    fetched_at: datetime | None = None,
) -> Manifest:
    """Assemble a ``Manifest`` from the current universe and artefact hashes.

    Args:
        window_start: Inclusive first date of the study window.
        window_end: Inclusive last date.
        artefacts: Mapping of ``manifest_key -> file_path``. Each file is
            hashed and the digest recorded under ``integrity[key]_sha256``.
            Files that do not exist are skipped.
        fetched_at: Timestamp of the fetch. Defaults to now (UTC).

    Returns:
        A populated ``Manifest``.
    """
    bucket_counts: dict[str, int] = {}
    for bucket in universe.BUCKETS:
        bucket_counts[bucket] = len(universe.by_bucket(bucket))

    integrity: dict[str, str] = {}
    for key, path in artefacts.items():
        if path.exists():
            integrity[f"{key}_sha256"] = compute_file_sha256(path)

    return Manifest(
        universe_version=UNIVERSE_VERSION,
        n_instruments=len(universe.UNIVERSE),
        bucket_counts=bucket_counts,
        window_start=window_start,
        window_end=window_end,
        fetched_at=fetched_at or datetime.now(UTC),
        integrity=integrity,
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Serialise a manifest to a YAML file.

    Uses block-style YAML (readable, git-diff-friendly) and preserves
    insertion order of keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest.to_dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def read_manifest(path: Path) -> dict[str, Any]:
    """Load an existing manifest file as a plain dict.

    Returns the parsed YAML as-is. Used by tests and for diagnostics; the
    pipeline itself doesn't read the manifest back.
    """
    with path.open("r", encoding="utf-8") as f:
        loaded: Any = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise TypeError(f"Manifest at {path} is not a mapping")
    return loaded
