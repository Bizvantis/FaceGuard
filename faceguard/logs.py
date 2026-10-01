"""Logging to a rotating file in the data folder (and optionally the console)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from faceguard import paths


def setup_logging(console: bool = True, level: int = logging.INFO) -> Path:
    log_file = paths.logs_dir() / "faceguard.log"
    handlers: list[logging.Handler] = [RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")]
    if console and sys.stderr is not None:  # pythonw has no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", handlers=handlers, force=True)
    return log_file
