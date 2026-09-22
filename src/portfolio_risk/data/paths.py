"""Filesystem locations for project data.

Provides constants pointing at the on-disk data cache and a helper to
ensure the cache subdirectories exist. All paths are resolved from the
project root, which is detected by walking up from this module's location
until a ``pyproject.toml`` is found.

The data cache (``data/raw/``, ``data/processed/``) is git-ignored per
the design document §6; code writes there freely and does not commit.
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing pyproject.toml is found.

    Args:
        start: Path to begin the search from. If a file, its parent is used.

    Returns:
        Absolute path of the first ancestor directory (inclusive of
        ``start``) that contains a ``pyproject.toml``.

    Raises:
        RuntimeError: If no such directory is found up to the filesystem root.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        f"Could not find pyproject.toml walking up from {start!r}. "
        "Are you running from inside the project checkout?"
    )


PROJECT_ROOT: Path = _find_project_root(Path(__file__))
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MANIFEST_PATH: Path = DATA_DIR / "manifest.yaml"


def ensure_data_dirs() -> None:
    """Create the standard data subdirectories if they don't exist.

    Idempotent — safe to call at the start of every acquisition or
    cleaning run.
    """
    for path in (RAW_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)
