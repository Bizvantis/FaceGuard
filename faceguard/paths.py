"""Filesystem locations used by FaceGuard.

User data (face database, settings, intruder snapshots, logs) lives in a per-user
folder so that it is protected by the Windows account:

* Windows: ``%LOCALAPPDATA%\\FaceGuard``
* other:   ``~/.faceguard``

Set ``FACEGUARD_HOME`` to use a different folder (the test-suite does this) and
``FACEGUARD_MODELS`` to load the neural-network models from somewhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    env = os.environ.get("FACEGUARD_HOME")
    if env:
        return _ensure(Path(env))
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return _ensure(base / "FaceGuard")
    return _ensure(Path.home() / ".faceguard")


def models_dir() -> Path:
    env = os.environ.get("FACEGUARD_MODELS")
    if env:
        return _ensure(Path(env))
    # Source checkout (the normal case): keep models next to the code.
    local = PROJECT_DIR / "models"
    if local.is_dir() or os.access(PROJECT_DIR, os.W_OK):
        return _ensure(local)
    return _ensure(data_dir() / "models")


def database_path() -> Path:
    return data_dir() / "faces.db"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def intruders_dir() -> Path:
    return _ensure(data_dir() / "intruders")


def snapshots_dir() -> Path:
    return _ensure(data_dir() / "snapshots")


def logs_dir() -> Path:
    return _ensure(data_dir() / "logs")
