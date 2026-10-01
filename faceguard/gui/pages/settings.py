from __future__ import annotations

import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QMessageBox, QSpinBox, QVBoxLayout, QWidget

from faceguard import APP_NAME, __version__, models, paths
from faceguard.config import SECURITY_LEVELS
from faceguard.gui import icons
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page, header, page_layout, scrolling
from faceguard.gui.widgets import SettingRow, Toggle, button, card, label, run_job
from faceguard.security import dpapi, winsys
from faceguard.vision.mesh import mediapipe_available

RESOLUTIONS = [((640, 480), "640 x 480 (recommended: highest frame rate)"), ((1280, 720), "1280 x 720 (sharper, often slower)"), ((1920, 1080), "1920 x 1080")]
LEVELS = [("relaxed", "Relaxed - fewer retries"), ("balanced", "Balanced (recommended)"), ("strict", "Strict - hardest to fool"),
          ("custom", "Custom threshold")]


class SettingsPage(Page):
    key = "settings"
    title = "Settings"
    glyph = icons.SETTINGS

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        s = ctx.settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content.setObjectName("Page")
        lay = page_layout(content)
        lay.addWidget(header("Settings"))

        # --- camera ---------------------------------------------------------
        cam, cl = card("Camera")
        self.cam_combo = QComboBox()
        for i in range(4):
            self.cam_combo.addItem(f"Camera {i}" + (" (default)" if i == 0 else ""), i)
        self.cam_combo.setCurrentIndex(max(0, min(3, s.camera_index)))
        detect = button("Detect cameras", None, icons.REFRESH, "Briefly opens each camera to see which ones work")
        detect.clicked.connect(self._detect)
        self.detect_btn = detect
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.cam_combo)
        rl.addWidget(detect)
        cl.addWidget(SettingRow("Which camera", "Laptops usually have one (camera 0)", row))
        self.res_combo = QComboBox()
        for size, text in RESOLUTIONS:
            self.res_combo.addItem(text, size)
        self.res_combo.setCurrentIndex(next((i for i, (sz, _) in enumerate(RESOLUTIONS) if sz == (s.camera_width, s.camera_height)), 0))
        cl.addWidget(SettingRow("Resolution", "Higher resolution recognises faces further away", self.res_combo))
        self.cam_combo.currentIndexChanged.connect(self._save)
        self.res_combo.currentIndexChanged.connect(self._save)
        lay.addWidget(cam)

        # --- recognition ------------------------------------------------------
        rec, rcl = card("Recognition")
        self.level = QComboBox()
        for key, text in LEVELS:
            self.level.addItem(text, key)
        self.level.setCurrentIndex(next((i for i, (k, _) in enumerate(LEVELS) if k == s.security_level), 1))
        self.level.setMinimumWidth(220)
        rcl.addWidget(SettingRow("Security level", "How similar a face must be to count as a match", self.level))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.30, 0.85)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(s.match_threshold)
        self.threshold_row = SettingRow("Match threshold", "Cosine similarity needed (the Model Lab can recommend one)", self.threshold)
        rcl.addWidget(self.threshold_row)
        self.adaptive = Toggle(s.adaptive_learning)
        rcl.addWidget(SettingRow("Adaptive learning", "Slowly learn your changing appearance from very confident matches", self.adaptive))
        self.level.currentIndexChanged.connect(self._level_changed)
        self.threshold.valueChanged.connect(self._save)
        self.adaptive.toggled.connect(self._save)
        lay.addWidget(rec)

        # --- liveness -----------------------------------------------------------
        live, ll = card("Liveness (anti-spoofing)")
        self.challenges = QSpinBox()
        self.challenges.setRange(1, 5)
        self.challenges.setValue(s.liveness_challenges)
        ll.addWidget(SettingRow("Number of actions", "More actions = harder to fake with a video, but slower", self.challenges))
        self.timeout = QSpinBox()
        self.timeout.setRange(4, 15)
        self.timeout.setSuffix(" s")
        self.timeout.setValue(int(s.liveness_timeout))
        ll.addWidget(SettingRow("Time per action", "", self.timeout))
        self.expr = Toggle(s.liveness_expressions and mediapipe_available())
        self.expr.setEnabled(mediapipe_available())
        ll.addWidget(SettingRow("Blink and smile challenges", "Uses the MediaPipe face mesh" +
                                ("" if mediapipe_available() else " (not installed)"), self.expr))
        self.challenges.valueChanged.connect(self._save)
        self.timeout.valueChanged.connect(self._save)
        self.expr.toggled.connect(self._save)
        lay.addWidget(live)

        # --- privacy -------------------------------------------------------------
        priv, pl = card("Privacy & data")
        self.encrypt = Toggle(ctx.db.encrypt)
        self.encrypt.setEnabled(dpapi.available())
        pl.addWidget(SettingRow("Encrypt face data", "Windows DPAPI: only your Windows account on this PC can read it", self.encrypt))
        self.encrypt.toggled.connect(self._toggle_encryption)
        open_data = button("Open data folder", None, icons.FOLDER)
        open_data.clicked.connect(lambda: winsys.open_path(paths.data_dir()))
        pl.addWidget(SettingRow("Data folder", str(paths.data_dir()), open_data))
        wipe = button("Delete all face data", "danger", icons.DELETE)
        wipe.clicked.connect(self._wipe)
        pl.addWidget(SettingRow("Start over", "Removes every enrolled person (snapshots are kept)", wipe))
        lay.addWidget(priv)

        # --- about -----------------------------------------------------------------
        about, al = card("About")
        import cv2

        try:
            import PySide6

            qt = PySide6.__version__
        except Exception:  # pragma: no cover
            qt = "?"
        al.addWidget(label(f"{APP_NAME} {__version__}  |  Python {platform.python_version()}  |  OpenCV {cv2.__version__}  |  "
                           f"PySide6 {qt}", "secondary"))
        for spec in models.MODELS.values():
            present = "installed" if models.is_present(spec.key) else "missing"
            al.addWidget(label(f"{spec.title}: {spec.source} ({spec.license}) - {present}", "muted", wrap=True))
        log_btn = button("Open log file", "ghost", icons.INFO)
        log_btn.clicked.connect(lambda: winsys.open_path(paths.logs_dir() / "faceguard.log"))
        al.addWidget(log_btn, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(about)
        lay.addStretch(1)
        outer.addWidget(scrolling(content))
        self._sync_level_ui()

    # ------------------------------------------------------------ handlers
    def _sync_level_ui(self) -> None:
        key = self.level.currentData()
        custom = key == "custom"
        self.threshold.setEnabled(custom)
        if not custom:
            self.threshold.blockSignals(True)
            self.threshold.setValue(SECURITY_LEVELS[key])
            self.threshold.blockSignals(False)

    def _level_changed(self) -> None:
        self._sync_level_ui()
        self._save()

    def _save(self) -> None:
        s = self.ctx.settings
        s.camera_index = int(self.cam_combo.currentData())
        s.camera_width, s.camera_height = self.res_combo.currentData()
        s.security_level = self.level.currentData()
        if s.security_level == "custom":
            s.custom_threshold = round(float(self.threshold.value()), 2)
        s.adaptive_learning = self.adaptive.isChecked()
        s.liveness_challenges = int(self.challenges.value())
        s.liveness_timeout = float(self.timeout.value())
        s.liveness_expressions = self.expr.isChecked()
        self.ctx.save_settings()

    def refresh_from_settings(self) -> None:
        s = self.ctx.settings
        self.level.blockSignals(True)
        self.level.setCurrentIndex(next((i for i, (k, _) in enumerate(LEVELS) if k == s.security_level), 1))
        self.level.blockSignals(False)
        self.threshold.blockSignals(True)
        self.threshold.setValue(s.match_threshold)
        self.threshold.setEnabled(s.security_level == "custom")
        self.threshold.blockSignals(False)

    def _detect(self) -> None:
        from faceguard.vision.camera import probe_cameras

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")

        def done(found):
            self.detect_btn.setEnabled(True)
            self.detect_btn.setText("Detect cameras")
            for i in range(self.cam_combo.count()):
                idx = self.cam_combo.itemData(i)
                match = next((d for j, d in found if j == idx), None)
                self.cam_combo.setItemText(i, f"Camera {idx}: " + (match or "not found"))
            self.ctx.toast.emit("info", f"Found {len(found)} working camera(s)")

        # The active camera is busy while in use; probing only sees cameras that are free.
        run_job(probe_cameras, on_done=done, on_error=lambda e: done([]))

    def _toggle_encryption(self, on: bool) -> None:
        self.ctx.settings.encrypt_database = on
        self.ctx.db.encrypt = on and dpapi.available()
        self.ctx.db.save()
        self.ctx.save_settings()
        self.ctx.toast.emit("good" if on else "warn", "Face data encrypted" if on else "Face data is now stored unencrypted")

    def _wipe(self) -> None:
        answer = QMessageBox.question(self, "Delete all face data",
                                      "Delete every enrolled person and their face data? This cannot be undone.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.db.delete_all()
            self.ctx.toast.emit("info", "All face data deleted")
