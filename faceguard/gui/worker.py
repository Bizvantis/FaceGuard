"""The camera / AI thread.

One QThread owns the camera, the FaceEngine (not thread-safe, so it lives
here only), the presence guard and any running task (enrollment or
verification).  The GUI talks to it through small thread-safe setters and
receives results through Qt signals (queued automatically across threads).

Privacy: the camera is opened only while something needs it - a camera page
is on screen, a task is running, or the guard is on - and it is released
while the PC is locked.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace

import cv2
from PySide6.QtCore import QThread, Signal

from faceguard.config import Settings
from faceguard.gui.widgets import bgr_to_qimage
from faceguard.recognition.adaptive import AdaptiveLearner
from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import EngineOptions, FaceEngine, FrameResult
from faceguard.recognition.enrollment import EnrollmentSession
from faceguard.security.guard import GuardState, guard_faces
from faceguard.security.liveness import LivenessConfig, VerificationTask
from faceguard.security.runner import GuardRunner
from faceguard.vision.camera import CameraError, open_source

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- tasks
class EnrollTask:
    kind = "enroll"
    needs_mesh = False

    def __init__(self, session: EnrollmentSession) -> None:
        self.session = session
        self.finished = False
        self._skip = False

    def skip_pose(self) -> None:  # called from the GUI thread
        self._skip = True

    def cancel(self) -> None:
        self.session.cancel()

    def update(self, frame, result: FrameResult, engine: FaceEngine, t: float):
        if self._skip:
            self._skip = False
            self.session.skip_current(t)
        state = self.session.update(frame, result, engine, t)
        self.finished = state.phase != "running"
        return state

    def outcome(self):
        return self.session.finish() if self.session.phase == "done" else None


class VerifyTask:
    kind = "verify"

    def __init__(self, config: LivenessConfig, use_expressions: bool, require_owner: bool) -> None:
        self.config = config
        self.use_expressions = use_expressions
        self.require_owner = require_owner
        self.finished = False
        self._task: VerificationTask | None = None
        self._cancelled = False

    @property
    def needs_mesh(self) -> bool:
        return self.use_expressions

    @property
    def sequence(self):
        return self._task.session.sequence if self._task else []

    def cancel(self) -> None:
        self._cancelled = True

    def update(self, frame, result: FrameResult, engine: FaceEngine, t: float):
        if self._task is None:
            expressions = self.use_expressions and engine.ensure_mesh()
            self._task = VerificationTask(engine.gallery, engine.matcher, self.config, expressions, self.require_owner)
        if self._cancelled:
            self.finished = True
            return None
        state = self._task.update(result)
        self.finished = state.finished
        return state

    def outcome(self):
        return None if self._cancelled or self._task is None else self._task.result


# ---------------------------------------------------------------- messages
@dataclass
class FramePacket:
    image: object  # QImage (display copy, already mirrored if requested)
    result: FrameResult
    task_kind: str | None
    task_state: object
    task_sequence: list
    fps: float
    mirrored: bool


@dataclass
class GuardStatus:
    state: GuardState
    text: str
    seconds_to_lock: float | None
    present: list[str]
    session_locked: bool


class VideoWorker(QThread):
    frame_ready = Signal(object)  # FramePacket
    camera_state = Signal(str, str)  # off | starting | on | error, message
    guard_status = Signal(object)  # GuardStatus
    notify = Signal(str, str, str)  # kind, title, message
    snapshot_saved = Signal(object)  # IntruderEvent
    task_finished = Signal(str, object, object)  # kind, task, outcome (None if cancelled)
    learned = Signal(str)  # adaptive learning added a sample for this person
    fatal = Signal(str)

    def __init__(self, settings: Settings, db: FaceDatabase, source_spec: str | int | None = None) -> None:
        super().__init__()
        self.setObjectName("FaceGuard-camera")
        self._db = db
        self._source_spec = source_spec
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._settings = replace(settings)
        self._settings_dirty = False
        self._stop = False
        self._view_active = False
        self._task = None
        self._guard_cmds: list[tuple] = []
        self._snapshot = False
        self._frame_in_flight = False
        self.camera_description = ""

    # ------------------------------------------------------------ control (GUI thread)
    def _poke(self) -> None:
        self._wake.set()

    def set_view_active(self, active: bool) -> None:
        with self._lock:
            self._view_active = active
        self._poke()

    def start_task(self, task) -> None:
        with self._lock:
            if self._task is not None:
                self._task.cancel()
            self._task = task
        self._poke()

    def cancel_task(self) -> None:
        with self._lock:
            if self._task is not None:
                self._task.cancel()
        self._poke()

    def guard_command(self, *cmd) -> None:
        """('start',) ('stop',) ('pause', seconds) ('resume',)"""
        with self._lock:
            self._guard_cmds.append(cmd)
        self._poke()

    def apply_settings(self, settings: Settings) -> None:
        with self._lock:
            self._settings = replace(settings)
            self._settings_dirty = True
        self._poke()

    def request_snapshot(self) -> None:
        with self._lock:
            self._snapshot = True

    def frame_consumed(self) -> None:
        self._frame_in_flight = False

    def stop(self) -> None:
        with self._lock:
            self._stop = True
        self._poke()
        self.wait(5000)

    # ------------------------------------------------------------ thread body
    def run(self) -> None:  # noqa: C901 - the main loop reads best in one place
        try:
            engine = FaceEngine.create(self._settings)
        except Exception as exc:
            log.exception("engine init failed")
            self.fatal.emit(f"Could not load the face-recognition models:\n{exc}")
            return
        runner = GuardRunner(self._settings, self._db, notify=self.notify.emit, on_snapshot=self.snapshot_saved.emit)
        learner = AdaptiveLearner(self._db)
        source = None
        source_key = None
        retry_at = 0.0
        last_processed = 0.0
        last_status = 0.0
        fps = 0.0
        last_frame_t: float | None = None
        gallery_version = -1

        def close_source():
            nonlocal source
            if source is not None:
                source.stop()
                source = None
                self.camera_state.emit("off", "")

        while True:
            with self._lock:
                if self._stop:
                    break
                s = self._settings
                dirty, self._settings_dirty = self._settings_dirty, False
                cmds, self._guard_cmds = self._guard_cmds, []
                view_active = self._view_active
                task = self._task
                snapshot, self._snapshot = self._snapshot, False
            now = time.monotonic()

            if dirty:
                engine.configure(s)
                runner.configure(s)
                key = (self._source_spec if self._source_spec is not None else s.camera_index, s.camera_width, s.camera_height)
                if source is not None and key != source_key:
                    close_source()
            for cmd in cmds:
                if cmd[0] == "start":
                    runner.start(now)
                elif cmd[0] == "stop":
                    runner.stop()
                elif cmd[0] == "pause":
                    runner.guard.pause(now, cmd[1])
                elif cmd[0] == "resume":
                    runner.guard.resume(now)
            guard_on = runner.state != GuardState.STOPPED

            session_locked = runner.poll_session(now) if guard_on else False
            if guard_on and now - last_status > 0.3:
                last_status = now
                g = runner.guard
                self.guard_status.emit(GuardStatus(g.state, g.status_text(now), g.seconds_to_lock, list(g.present_names), session_locked))
            if session_locked and not runner.guard.watching_locked(now):
                close_source()  # nobody can use a locked PC: release the camera until it is unlocked
                self._wake.wait(0.5)
                self._wake.clear()
                continue

            guard_watching = guard_on and runner.state not in (GuardState.PAUSED,)
            if not (view_active or task is not None or guard_watching):
                close_source()
                self._wake.wait(0.25)
                self._wake.clear()
                continue

            if source is None:
                if now < retry_at:
                    self._wake.wait(min(0.25, retry_at - now))
                    self._wake.clear()
                    continue
                spec = self._source_spec if self._source_spec is not None else s.camera_index
                self.camera_state.emit("starting", "Opening camera...")
                try:
                    source = open_source(spec, s.camera_width, s.camera_height).start()
                    source_key = (spec, s.camera_width, s.camera_height)
                    self.camera_description = f"{getattr(source, 'backend', '') or source.name}"
                    runner.guard.on_camera_restored(time.monotonic())
                    self.camera_state.emit("on", self.camera_description)
                except CameraError as exc:
                    self.camera_state.emit("error", str(exc))
                    runner.guard.on_camera_lost(now)
                    retry_at = now + 5.0
                    continue

            frame, ts = source.read(1.0)
            if frame is None:
                if source.error:
                    self.camera_state.emit("error", source.error)
                    source.stop()
                    source = None
                    runner.guard.on_camera_lost(now)
                    retry_at = now + 3.0
                continue

            low_power = session_locked or (guard_watching and not view_active and task is None)
            if low_power:
                wait = 1.0 / max(0.5, s.guard_fps) - (ts - last_processed)
                if wait > 0:
                    self._wake.wait(min(wait, 0.25))
                    self._wake.clear()
                    continue
            last_processed = ts

            if self._db.version != gallery_version:
                engine.set_gallery(self._db.snapshot())
                gallery_version = self._db.version
            mesh = bool((task is not None and task.needs_mesh) or (view_active and s.show_mesh))
            try:
                result = engine.process(frame, ts, EngineOptions(mesh=mesh, embed_interval=0.2 if low_power else 0.35))
            except Exception:
                log.exception("frame processing failed")
                continue

            if session_locked:  # lock-screen watch: only look for strangers, nothing else runs
                runner.process_locked(frame, result, ts)
                continue

            task_state = None
            if task is not None:
                try:
                    task_state = task.update(frame, result, engine, ts)
                except Exception:
                    log.exception("task failed")
                    task.finished = True
                if task.finished:
                    with self._lock:
                        if self._task is task:
                            self._task = None
                    self.task_finished.emit(task.kind, task, task.outcome())

            if guard_watching and runner.guard.active:
                runner.process(frame, result, ts)
            if s.adaptive_learning and (runner.state == GuardState.PRESENT or view_active) and task is None:
                name = learner.observe(result, engine.gallery, engine.matcher.threshold)
                if name:
                    self.learned.emit(name)
            if snapshot:
                boxes = [(f.bbox, f.match.label, f.match.is_match) for f in result.faces]
                try:
                    self.snapshot_saved.emit(runner.intruders.record(frame, "manual", guard_faces(result), boxes))
                except OSError:
                    log.exception("snapshot failed")

            if last_frame_t is not None:
                dt = max(1e-3, ts - last_frame_t)
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            last_frame_t = ts
            if (view_active or task is not None) and not self._frame_in_flight:
                display = cv2.flip(frame, 1) if s.mirror_preview else frame
                self._frame_in_flight = True
                self.frame_ready.emit(FramePacket(
                    bgr_to_qimage(display), result, task.kind if task else None, task_state,
                    list(getattr(task, "sequence", []) or []), fps, s.mirror_preview,
                ))

        close_source()
        engine.close()
