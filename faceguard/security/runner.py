"""Glue between the guard's decisions and the real world (Windows lock, snapshots, notifications)."""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from faceguard.config import Settings
from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import FrameResult
from faceguard.security import winsys
from faceguard.security.guard import GuardAction, GuardConfig, GuardState, PresenceGuard, guard_faces
from faceguard.security.intruders import IntruderEvent, IntruderLog

log = logging.getLogger(__name__)
Notifier = Callable[[str, str, str], None]  # (kind, title, message)


def config_from_settings(s: Settings) -> GuardConfig:
    return GuardConfig(
        lock_on_absence=s.guard_lock_on_absence,
        absence_timeout=float(s.guard_absence_timeout),
        respect_activity=s.guard_respect_activity,
        lock_on_intruder=s.guard_lock_on_intruder,
        intruder_seconds=float(s.guard_intruder_seconds),
        save_intruders=s.guard_save_intruders,
        shoulder_alert=s.guard_shoulder_alert,
        watch_locked_minutes=float(s.guard_locked_watch_minutes),
    )


class GuardRunner:
    def __init__(
        self,
        settings: Settings,
        db: FaceDatabase,
        notify: Notifier | None = None,
        lock_fn: Callable[[], bool] = winsys.lock_workstation,
        intruder_log: IntruderLog | None = None,
        on_snapshot: Callable[[IntruderEvent], None] | None = None,
    ) -> None:
        self.guard = PresenceGuard(config_from_settings(settings))
        self.db = db
        self.notify = notify or (lambda kind, title, msg: log.info("%s: %s - %s", kind, title, msg))
        self.lock_fn = lock_fn
        self.intruders = intruder_log or IntruderLog()
        self.on_snapshot = on_snapshot
        self._last_session_check = -1e9
        self._session_locked = False

    def configure(self, settings: Settings) -> None:
        self.guard.cfg = config_from_settings(settings)

    # ------------------------------------------------------------ lifecycle
    def start(self, t: float | None = None) -> None:
        self.guard.start(time.monotonic() if t is None else t)

    def stop(self) -> None:
        self.guard.stop()

    @property
    def state(self) -> GuardState:
        return self.guard.state

    def status_text(self, t: float | None = None) -> str:
        return self.guard.status_text(t)

    def poll_session(self, t: float) -> bool:
        """Track the Windows lock state (at most twice a second). Returns True while locked."""
        if t - self._last_session_check < 0.5:
            return self._session_locked
        self._last_session_check = t
        locked = bool(winsys.is_session_locked())
        if locked and not self._session_locked:
            self.guard.on_session_locked(t)
        elif not locked and self._session_locked:
            self.guard.on_session_unlocked(t)
        self._session_locked = locked
        return locked

    # ------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, result: FrameResult, t: float) -> list[GuardAction]:
        owner = self.db.owner()
        owner_enrolled = owner is not None and owner.count() > 0
        actions = self.guard.update(t, guard_faces(result), winsys.idle_seconds(), owner_enrolled)
        for action in actions:
            try:
                self._execute(action, frame, result)
            except Exception:  # an action failing must never kill the camera loop
                log.exception("guard action %s failed", action.kind)
        return actions

    def process_locked(self, frame: np.ndarray, result: FrameResult, t: float) -> list[GuardAction]:
        """Lock-screen watch: record anyone unrecognised at the PC while it is locked."""
        actions = self.guard.update_locked(t, guard_faces(result))
        for action in actions:
            try:
                self._execute(action, frame, result)
            except Exception:
                log.exception("guard action %s failed", action.kind)
        return actions

    def _execute(self, action: GuardAction, frame: np.ndarray, result: FrameResult) -> None:
        if action.kind == "lock":
            log.warning("Locking workstation (%s)", action.reason)
            self.lock_fn()
        elif action.kind == "snapshot":
            boxes = [(f.bbox, f.match.label if f.match.person or f.match.is_unknown else "?",
                      bool(f.match.is_match and f.match.person and f.match.person.authorized)) for f in result.faces]
            event = self.intruders.record(frame, action.reason, list(action.faces), boxes)
            if self.on_snapshot:
                self.on_snapshot(event)
        elif action.kind in ("notify", "warn"):
            self.notify(action.reason, action.title, action.message)
