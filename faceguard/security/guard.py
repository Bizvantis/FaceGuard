"""Presence guard: the decision logic behind auto-lock and intruder detection.

This module is pure logic (no camera, no Windows calls) so it can be unit
tested with simulated timelines.  The camera worker feeds it one ``update``
per processed frame and executes the returned actions (lock the PC, save a
snapshot, show a notification).

Rules
-----
* **Walk-away lock** - no authorised face (owner or trusted person) for
  ``absence_timeout`` seconds *and* no keyboard/mouse input for that long.
  A warning is issued a few seconds before locking.
* **Intruder** - a face that is confidently *not* authorised, close to the
  screen and of decent quality, persists for ``intruder_seconds`` while no
  authorised person is present -> snapshot + notification + lock.
* **Shoulder surfing** - an authorised user is present and another,
  unauthorised face is watching for ``shoulder_seconds`` -> notification.
* After you unlock the PC there is a grace period, and intruder locks are
  rate-limited, so a recognition problem can never lock you out in a loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from faceguard.recognition.engine import FrameResult


class GuardState(str, Enum):
    STOPPED = "stopped"
    NO_OWNER = "no_owner"
    PRESENT = "present"
    WATCHING = "watching"
    INTRUDER = "intruder"
    LOCKED = "locked"
    PAUSED = "paused"
    NO_CAMERA = "no_camera"


@dataclass
class GuardConfig:
    lock_on_absence: bool = True
    absence_timeout: float = 30.0
    respect_activity: bool = True
    lock_on_intruder: bool = True
    intruder_seconds: float = 2.5
    save_intruders: bool = True
    shoulder_alert: bool = True
    shoulder_seconds: float = 3.0
    warn_before_lock: float = 8.0
    grace_after_unlock: float = 15.0
    alert_cooldown: float = 60.0
    min_intruder_face_px: float = 60.0
    min_watcher_face_px: float = 40.0
    max_intruder_locks: int = 3
    intruder_window: float = 600.0
    lock_retry: float = 5.0
    watch_locked_minutes: float = 30.0  # keep watching this long after the PC locks (0 = off)


@dataclass(frozen=True)
class GuardFace:
    track_id: int
    authorized: bool  # recognised as the owner or a trusted person
    stranger: bool  # confidently NOT authorised (unknown, or enrolled but not trusted)
    name: str | None
    similarity: float
    face_px: float
    quality_ok: bool


@dataclass(frozen=True)
class GuardAction:
    kind: str  # lock | snapshot | notify | warn
    reason: str
    title: str = ""
    message: str = ""
    faces: tuple[GuardFace, ...] = ()


def guard_faces(result: FrameResult) -> list[GuardFace]:
    """Translate engine output into the guard's view of each face."""
    out = []
    for f in result.faces:
        m = f.match
        person = m.person
        authorized = m.is_match and person is not None and person.authorized
        # Needs >= 2 embeddings so one bad frame can never raise an alarm.
        stranger = f.embed_count >= 2 and (m.is_unknown or (m.is_match and person is not None and not person.authorized))
        out.append(
            GuardFace(
                track_id=f.track_id,
                authorized=authorized,
                stranger=stranger,
                name=person.name if (m.is_match and person) else None,
                similarity=m.similarity,
                face_px=f.face_px,
                quality_ok=f.quality is not None and f.quality.ok,
            )
        )
    return out


@dataclass
class _Timer:
    since: dict[int, float] = field(default_factory=dict)

    def update(self, ids: set[int], t: float) -> dict[int, float]:
        self.since = {i: self.since.get(i, t) for i in ids}
        return {i: t - s for i, s in self.since.items()}

    def clear(self) -> None:
        self.since.clear()


class PresenceGuard:
    def __init__(self, config: GuardConfig | None = None) -> None:
        self.cfg = config or GuardConfig()
        self.state = GuardState.STOPPED
        self.last_authorized = 0.0
        self.seconds_to_lock: float | None = None
        self.present_names: list[str] = []
        self._grace_until = 0.0
        self._warned = False
        self._lock_requested = -1e9
        self._last_intruder_alert = -1e9
        self._last_shoulder_alert = -1e9
        self._intruder_locks: deque[float] = deque()
        self._rate_limit_notified = False
        self._unrecognized_notified = False
        self._paused_until: float | None = None
        self._locked_since: float | None = None
        self._last_locked_alert = -1e9
        self._strangers = _Timer()
        self._watchers = _Timer()

    # ------------------------------------------------------------ lifecycle
    def start(self, t: float) -> None:
        self.state = GuardState.WATCHING
        self.last_authorized = t  # assume the user is at the PC when they switch the guard on
        self._grace_until = t + 5.0
        self._reset_timers()

    def stop(self) -> None:
        self.state = GuardState.STOPPED
        self.seconds_to_lock = None

    @property
    def active(self) -> bool:
        return self.state not in (GuardState.STOPPED, GuardState.PAUSED)

    def pause(self, t: float, seconds: float | None = None) -> None:
        self.state = GuardState.PAUSED
        self._paused_until = t + seconds if seconds else None
        self.seconds_to_lock = None

    def resume(self, t: float) -> None:
        self.start(t)
        self._paused_until = None

    def on_session_locked(self, t: float) -> None:
        if self.state not in (GuardState.STOPPED, GuardState.PAUSED):  # a pause outlives a lock/unlock
            self.state = GuardState.LOCKED
            self.seconds_to_lock = None
            self._locked_since = t
            self._strangers.clear()

    def on_session_unlocked(self, t: float) -> None:
        self._locked_since = None
        if self.state == GuardState.LOCKED:
            self.state = GuardState.WATCHING
            self.last_authorized = t
            self._grace_until = t + self.cfg.grace_after_unlock
            self._unrecognized_notified = False
            self._reset_timers()

    def watching_locked(self, t: float) -> bool:
        """True while the PC is locked and the lock-screen watch window is still open."""
        return (
            self.state == GuardState.LOCKED
            and self.cfg.watch_locked_minutes > 0
            and self._locked_since is not None
            and t - self._locked_since <= self.cfg.watch_locked_minutes * 60.0
        )

    def update_locked(self, t: float, faces: list[GuardFace]) -> list[GuardAction]:
        """While locked: photograph anyone unrecognised who sits at the PC (no lock needed - it is locked)."""
        if not self.watching_locked(t):
            return []
        cfg = self.cfg
        close = [f for f in faces if f.stranger and f.quality_ok and f.face_px >= cfg.min_intruder_face_px]
        age = self._strangers.update({f.track_id for f in close}, t)
        confirmed = [f for f in close if age[f.track_id] >= cfg.intruder_seconds]
        if not confirmed or t - self._last_locked_alert < cfg.alert_cooldown:
            return []
        self._last_locked_alert = t
        who = ", ".join(f.name for f in confirmed if f.name) or "an unknown person"
        actions = []
        if cfg.save_intruders:
            actions.append(GuardAction("snapshot", "locked", faces=tuple(faces)))
        actions.append(GuardAction("notify", "locked", "Someone was at your locked PC",
                                   f"FaceGuard saw {who} at your PC while it was locked.", tuple(confirmed)))
        return actions

    def on_camera_lost(self, t: float) -> None:
        if self.active and self.state != GuardState.LOCKED:
            self.state = GuardState.NO_CAMERA
            self.seconds_to_lock = None

    def on_camera_restored(self, t: float) -> None:
        if self.state == GuardState.NO_CAMERA:
            self.start(t)  # never lock immediately because of time spent without a camera

    def _reset_timers(self) -> None:
        self._warned = False
        self._strangers.clear()
        self._watchers.clear()
        self.seconds_to_lock = None

    # ------------------------------------------------------------ decision
    def update(self, t: float, faces: list[GuardFace], idle_seconds: float, owner_enrolled: bool) -> list[GuardAction]:
        if self.state == GuardState.PAUSED:
            if self._paused_until is not None and t >= self._paused_until:
                self.resume(t)
            else:
                return []
        if self.state in (GuardState.STOPPED, GuardState.LOCKED, GuardState.NO_CAMERA):
            return []
        if not owner_enrolled:
            self.state = GuardState.NO_OWNER
            self.seconds_to_lock = None
            return []
        if self.state == GuardState.NO_OWNER:
            self.start(t)

        cfg = self.cfg
        actions: list[GuardAction] = []
        authorized = [f for f in faces if f.authorized]
        self.present_names = sorted({f.name for f in authorized if f.name})
        if authorized:
            self.last_authorized = t
            self._warned = False

        close_strangers = [f for f in faces if f.stranger and f.quality_ok and f.face_px >= cfg.min_intruder_face_px]
        watchers = [f for f in faces if f.stranger and f.face_px >= cfg.min_watcher_face_px]
        stranger_age = self._strangers.update({f.track_id for f in close_strangers}, t)
        watcher_age = self._watchers.update({f.track_id for f in watchers}, t)
        confirmed = [f for f in close_strangers if stranger_age[f.track_id] >= cfg.intruder_seconds]
        in_grace = t < self._grace_until
        authorized_recent = t - self.last_authorized < 3.0

        # --- intruder at the keyboard -----------------------------------------
        if confirmed and not authorized_recent:
            if in_grace:
                if not self._unrecognized_notified:
                    self._unrecognized_notified = True
                    actions.append(GuardAction(
                        "notify", "unrecognized", "FaceGuard didn't recognise you",
                        "If this is you, open FaceGuard and add more face samples to improve recognition.",
                    ))
                return actions
            self.state = GuardState.INTRUDER
            self.seconds_to_lock = None
            if t - self._last_intruder_alert >= cfg.alert_cooldown:
                self._last_intruder_alert = t
                who = ", ".join(f.name for f in confirmed if f.name) or "an unknown person"
                if cfg.save_intruders:
                    actions.append(GuardAction("snapshot", "intruder", faces=tuple(faces)))
                actions.append(GuardAction("notify", "intruder", "Intruder detected",
                                           f"FaceGuard saw {who} at your PC.", tuple(confirmed)))
                if cfg.lock_on_intruder:
                    while self._intruder_locks and t - self._intruder_locks[0] > cfg.intruder_window:
                        self._intruder_locks.popleft()
                    if len(self._intruder_locks) < cfg.max_intruder_locks:
                        self._intruder_locks.append(t)
                        self._lock_requested = t
                        actions.append(GuardAction("lock", "intruder", faces=tuple(confirmed)))
                    elif not self._rate_limit_notified:
                        self._rate_limit_notified = True
                        actions.append(GuardAction(
                            "notify", "rate_limited", "Intruder lock paused",
                            "FaceGuard locked the PC several times in a few minutes. If that was you, "
                            "re-enroll your face in better light.",
                        ))
            return actions

        # --- someone looking over the user's shoulder ----------------------------
        if cfg.shoulder_alert and authorized:
            lurkers = [f for f in watchers if watcher_age[f.track_id] >= cfg.shoulder_seconds]
            if lurkers and t - self._last_shoulder_alert >= cfg.alert_cooldown:
                self._last_shoulder_alert = t
                actions.append(GuardAction("notify", "shoulder", "Someone is looking at your screen",
                                           "FaceGuard noticed an unrecognised person behind you.", tuple(lurkers)))
                if cfg.save_intruders:
                    actions.append(GuardAction("snapshot", "shoulder", faces=tuple(faces)))

        if authorized:
            self.state = GuardState.PRESENT
            self.seconds_to_lock = None
            return actions

        # --- nobody authorised in view: walk-away countdown ---------------------
        self.state = GuardState.WATCHING
        if not cfg.lock_on_absence:
            self.seconds_to_lock = None
            return actions
        away = t - self.last_authorized
        if cfg.respect_activity:
            away = min(away, idle_seconds)  # typing/mouse use means someone is at the PC
        remaining = cfg.absence_timeout - away
        if in_grace:
            remaining = max(remaining, self._grace_until - t)
        self.seconds_to_lock = max(0.0, remaining)
        if remaining <= 0:
            if t - self._lock_requested >= cfg.lock_retry:
                self._lock_requested = t
                actions.append(GuardAction("lock", "absence"))
        elif remaining <= cfg.warn_before_lock and not self._warned:
            self._warned = True
            actions.append(GuardAction("warn", "absence", "Locking soon",
                                       f"No one recognised - locking in {int(remaining + 0.5)} s. Look at the camera to stay unlocked."))
        return actions

    # ------------------------------------------------------------ presentation
    def status_text(self, t: float | None = None) -> str:
        s = self.state
        if s == GuardState.STOPPED:
            return "Guard is off"
        if s == GuardState.PAUSED:
            return "Paused"
        if s == GuardState.NO_OWNER:
            return "No owner enrolled - enroll yourself first"
        if s == GuardState.NO_CAMERA:
            return "Camera unavailable - retrying"
        if s == GuardState.LOCKED:
            return "PC locked - watching for intruders" if t is not None and self.watching_locked(t) else "PC locked"
        if s == GuardState.INTRUDER:
            return "Intruder detected!"
        if s == GuardState.PRESENT:
            return "Protecting " + (", ".join(self.present_names) or "you")
        if self.seconds_to_lock is not None and self.cfg.lock_on_absence:
            return f"No one recognised - locking in {int(self.seconds_to_lock + 0.5)} s"
        return "Watching"
