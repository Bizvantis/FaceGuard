from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext, face_color, face_title
from faceguard.gui.pages.base import Page
from faceguard.gui.widgets import (
    Meter, Pill, SettingRow, StatTile, Toggle, VideoView, button, card, label, rounded_pixmap,
)

MAX_ROWS = 4


class FaceRow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(46, 46)
        lay.addWidget(self.thumb)
        col = QVBoxLayout()
        col.setSpacing(3)
        top = QHBoxLayout()
        self.name = label("", "h3")
        self.pill = Pill()
        top.addWidget(self.name, 1)
        top.addWidget(self.pill)
        col.addLayout(top)
        self.meter = Meter()
        col.addWidget(self.meter)
        self.detail = label("", "small")
        col.addWidget(self.detail)
        lay.addLayout(col, 1)

    def show_face(self, f, thumb, threshold: float) -> None:
        title, _ = face_title(f)
        self.name.setText(title)
        color = face_color(f)
        status = {"match": "Recognised", "unknown": "Unknown", "uncertain": "Not sure", "pending": "Scanning"}[f.match.status]
        if f.match.is_match and f.match.person and not f.match.person.authorized:
            status = "Not trusted"
        self.pill.set_state(status, color)
        self.meter.set(max(0.0, f.match.similarity), color, marker=threshold)
        q = f.quality
        quality = f"quality {q.score:.0%}" if q else "quality -"
        issue = f" - {q.issues[0]}" if (q and q.issues) else ""
        self.detail.setText(f"similarity {f.match.similarity:.3f} (threshold {threshold:.2f})  |  {quality}{issue}")
        self.thumb.setPixmap(rounded_pixmap(thumb, 46, 10))


class LivePage(Page):
    key = "live"
    title = "Live"
    glyph = icons.VIDEO
    uses_camera = True

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        self._thumbs: dict[int, tuple[float, object]] = {}
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        self.video = VideoView()
        self.video.clear("Starting camera...", "")
        root.addWidget(self.video, 1)

        side = QVBoxLayout()
        side.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(label("Live recognition", "h1"), 1)
        self.cam_pill = Pill("Camera off", theme.MUTED)
        head.addWidget(self.cam_pill, 0, Qt.AlignmentFlag.AlignVCenter)
        side.addLayout(head)

        tiles = QGridLayout()
        tiles.setSpacing(10)
        self.t_fps = StatTile("Frame rate", "-", "frames per second")
        self.t_faces = StatTile("Faces", "0", "in view")
        self.t_lat = StatTile("Latency", "-", "detect + recognise")
        tiles.addWidget(self.t_fps, 0, 0)
        tiles.addWidget(self.t_faces, 0, 1)
        tiles.addWidget(self.t_lat, 0, 2)
        side.addLayout(tiles)

        self.empty_card, empty_lay = card("Nobody is enrolled yet",
                                          "Enroll your face so FaceGuard can recognise you. It takes about 20 seconds.")
        go = button("Enroll my face", "primary", icons.ADD_PERSON)
        go.clicked.connect(lambda: self.ctx.navigate.emit("enroll"))
        empty_lay.addWidget(go, 0, Qt.AlignmentFlag.AlignLeft)
        side.addWidget(self.empty_card)

        faces_card, faces_lay = card("In view")
        self.rows = [FaceRow() for _ in range(MAX_ROWS)]
        for r in self.rows:
            faces_lay.addWidget(r)
            r.hide()
        self.none_label = label("No faces detected", "muted")
        faces_lay.addWidget(self.none_label)
        side.addWidget(faces_card)

        ov_card, ov_lay = card("Overlays")
        s = ctx.settings
        self.tg_land = Toggle(s.show_landmarks)
        self.tg_mesh = Toggle(s.show_mesh)
        self.tg_pose = Toggle(s.show_pose)
        self.tg_mirror = Toggle(s.mirror_preview)
        ov_lay.addWidget(SettingRow("Facial landmarks", "The 5 points used to align each face", self.tg_land))
        ov_lay.addWidget(SettingRow("AI face mesh", "478-point 3-D mesh (MediaPipe)", self.tg_mesh))
        ov_lay.addWidget(SettingRow("Head pose", "Where the face is pointing", self.tg_pose))
        ov_lay.addWidget(SettingRow("Mirror preview", "Show the image like a mirror", self.tg_mirror))
        for t in (self.tg_land, self.tg_mesh, self.tg_pose, self.tg_mirror):
            t.toggled.connect(self._overlay_changed)
        side.addWidget(ov_card)

        snap = button("Save snapshot", None, icons.CAMERA, "Save the current frame to the snapshots gallery")
        snap.clicked.connect(self._snapshot)
        side.addWidget(snap, 0, Qt.AlignmentFlag.AlignLeft)
        side.addStretch(1)

        wrap = QWidget()
        wrap.setLayout(side)
        wrap.setFixedWidth(380)
        root.addWidget(wrap)
        ctx.db_changed.connect(self._refresh_empty)
        self._refresh_empty()

    def _refresh_empty(self) -> None:
        self.empty_card.setVisible(not any(p.samples for p in self.ctx.db.persons))

    def _overlay_changed(self) -> None:
        s = self.ctx.settings
        s.show_landmarks, s.show_mesh = self.tg_land.isChecked(), self.tg_mesh.isChecked()
        s.show_pose, s.mirror_preview = self.tg_pose.isChecked(), self.tg_mirror.isChecked()
        self.ctx.save_settings()

    def _snapshot(self) -> None:
        self.ctx.worker.request_snapshot()
        self.ctx.toast.emit("good", "Snapshot saved - see the Intruders & snapshots page")

    def set_camera_state(self, state: str, message: str) -> None:
        if state == "on":
            self.cam_pill.set_state("Camera on", theme.GOOD)
        elif state == "starting":
            self.cam_pill.set_state("Starting", theme.ACCENT)
            self.video.clear("Starting camera...")
        elif state == "error":
            self.cam_pill.set_state("Camera error", theme.CRITICAL)
            self.video.clear("Camera unavailable", message)
        else:
            self.cam_pill.set_state("Camera off", theme.MUTED)

    def on_frame(self, packet) -> None:
        r = packet.result
        now = time.monotonic()
        self.video.set_frame(packet.image, self.ctx.overlays(r), packet.mirrored)
        self.t_fps.set(f"{packet.fps:.0f}")
        self.t_faces.set(str(len(r.faces)))
        self.t_lat.set(f"{r.timings.get('total_ms', 0):.0f} ms",
                       f"detect {r.timings.get('detect_ms', 0):.0f} / embed {r.timings.get('embed_ms', 0):.0f}"
                       + (f" / mesh {r.timings['mesh_ms']:.0f}" if r.timings.get("mesh_ms", 0) > 0.5 else "") + " ms")
        for f in r.faces:
            if f.aligned is not None:
                self._thumbs[f.track_id] = (now, f.aligned)
        self._thumbs = {k: v for k, v in self._thumbs.items() if now - v[0] < 5.0}
        threshold = self.ctx.settings.match_threshold
        for i, row in enumerate(self.rows):
            if i < len(r.faces):
                f = r.faces[i]
                row.show_face(f, self._thumbs.get(f.track_id, (0, None))[1], threshold)
                row.show()
            else:
                row.hide()
        self.none_label.setVisible(not r.faces)
