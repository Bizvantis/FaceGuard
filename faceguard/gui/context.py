"""Shared application state handed to every page."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from faceguard.config import Settings
from faceguard.gui import theme
from faceguard.gui.widgets import FaceBox
from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import FaceResult, FrameResult
from faceguard.security.intruders import IntruderLog


def face_color(f: FaceResult) -> str:
    """Status colour for a face box (always shown together with a text label)."""
    m = f.match
    if m.is_match:
        return theme.GOOD if (m.person and m.person.authorized) else theme.SERIOUS
    if m.is_unknown:
        return theme.CRITICAL
    if m.status == "uncertain":
        return theme.WARNING
    return theme.ACCENT


def face_title(f: FaceResult) -> tuple[str, str]:
    m = f.match
    if m.status == "pending":
        return "Scanning...", ""
    title = m.label
    if m.is_match and m.person:
        title = m.person.name + ("  ★" if m.person.is_owner else "")
    return title, f"{m.similarity:.2f}"


class AppContext(QObject):
    db_changed = Signal()
    settings_changed = Signal()
    intruders_changed = Signal()
    navigate = Signal(str)
    enroll_person = Signal(str)  # open the Enroll page for this name
    toast = Signal(str, str)  # level (info|good|warn|error), message

    def __init__(self, settings: Settings, db: FaceDatabase, worker, intruders: IntruderLog) -> None:
        super().__init__()
        self.settings = settings
        self.db = db
        self.worker = worker
        self.intruders = intruders
        self.guard_status = None
        db.subscribe(self.db_changed.emit)  # may fire from the camera thread: Qt queues it to the GUI

    def save_settings(self) -> None:
        self.settings.save()
        self.worker.apply_settings(self.settings)
        self.settings_changed.emit()

    def overlays(self, result: FrameResult, landmarks: bool | None = None, mesh: bool | None = None,
                 pose: bool | None = None, titles: bool = True) -> list[FaceBox]:
        s = self.settings
        landmarks = s.show_landmarks if landmarks is None else landmarks
        mesh = s.show_mesh if mesh is None else mesh
        pose = s.show_pose if pose is None else pose
        boxes = []
        for f in result.faces:
            title, sub = face_title(f) if titles else ("", "")
            boxes.append(FaceBox(
                rect=tuple(float(v) for v in f.bbox), color=face_color(f), title=title, subtitle=sub,
                landmarks=f.landmarks if landmarks else None,
                mesh=f.mesh.points if (mesh and f.mesh is not None) else None,
                pose=(f.pose.yaw, f.pose.pitch) if pose else None,
            ))
        return boxes
