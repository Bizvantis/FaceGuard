"""User settings, persisted as JSON in the data folder."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from faceguard import paths

log = logging.getLogger(__name__)

# Cosine-similarity acceptance thresholds for the SFace embedding.
# OpenCV's reference operating point on LFW is 0.363; FaceGuard uses a top-k mean
# over several enrolled samples plus temporal smoothing, and defaults a little
# stricter because a false accept matters more than a retry.
SECURITY_LEVELS: dict[str, float] = {
    "relaxed": 0.38,
    "balanced": 0.45,
    "strict": 0.52,
}


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    camera_width: int = 640  # many laptop webcams deliver only ~10 fps at 1280x720; liveness needs the frame rate
    camera_height: int = 480
    mirror_preview: bool = True

    # Recognition
    security_level: str = "balanced"  # relaxed | balanced | strict | custom
    custom_threshold: float = 0.45
    unknown_threshold: float = 0.30  # below this a face is confidently "not enrolled"
    detection_score: float = 0.80
    min_face_size: int = 40  # pixels (full-resolution frame)
    top_k: int = 3
    adaptive_learning: bool = True

    # Live view overlays
    show_landmarks: bool = True
    show_mesh: bool = False
    show_pose: bool = False

    # Liveness
    liveness_challenges: int = 3
    liveness_timeout: float = 7.0
    liveness_expressions: bool = True  # blink / smile challenges (needs MediaPipe)

    # Presence guard
    guard_enabled: bool = False
    guard_lock_on_absence: bool = True
    guard_absence_timeout: int = 30
    guard_respect_activity: bool = True
    guard_lock_on_intruder: bool = True
    guard_intruder_seconds: float = 2.5
    guard_save_intruders: bool = True
    guard_shoulder_alert: bool = True
    guard_locked_watch_minutes: int = 30  # photograph people at the locked PC for this long (0 = off)
    guard_fps: float = 4.0

    # App behaviour
    minimize_to_tray: bool = True
    start_minimized: bool = False
    encrypt_database: bool = True
    intruder_retention_days: int = 30

    # ------------------------------------------------------------------ helpers
    @property
    def match_threshold(self) -> float:
        if self.security_level == "custom":
            return float(self.custom_threshold)
        return SECURITY_LEVELS.get(self.security_level, SECURITY_LEVELS["balanced"])

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or paths.settings_path()
        settings = cls()
        if not path.exists():
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read settings (%s); using defaults", exc)
            return settings
        known = {f.name: f for f in fields(cls)}
        for key, value in data.items():
            if key not in known:
                continue
            default = getattr(settings, key)
            try:
                # Coerce to the default's type so a hand-edited file cannot inject odd types.
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
                elif isinstance(default, str):
                    value = str(value)
            except (TypeError, ValueError):
                continue
            setattr(settings, key, value)
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or paths.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
