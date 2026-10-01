"""Smoke test of the desktop app: real main window, real camera thread, still-image 'camera'."""

import os
import time

import pytest

pytest.importorskip("PySide6")

from tests.conftest import DATA, needs_models  # noqa: E402

pytestmark = needs_models


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_app_processes_frames_and_every_page_opens(qapp, monkeypatch):
    import faceguard.security.winsys as ws

    monkeypatch.setattr(ws, "lock_workstation", lambda: True)  # never lock the machine running the tests
    from PySide6.QtCore import QThreadPool

    from faceguard.config import Settings
    from faceguard.gui.context import AppContext
    from faceguard.gui.main_window import MainWindow
    from faceguard.gui.worker import VideoWorker
    from faceguard.recognition.database import FaceDatabase
    from faceguard.security.intruders import IntruderLog

    settings = Settings()
    db = FaceDatabase.open()
    worker = VideoWorker(settings, db, str(DATA / "astronaut.jpg"))
    ctx = AppContext(settings, db, worker, IntruderLog())
    win = MainWindow(ctx)
    frames = []
    worker.frame_ready.connect(frames.append)
    worker.start()
    win.show()
    deadline = time.monotonic() + 30
    while len(frames) < 3 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    for key in ("live", "enroll", "people", "verify", "guard", "intruders", "lab", "settings"):
        win.go(key)
        qapp.processEvents()
        assert win.stack.currentWidget() is win.by_key[key]
    win.quit_app()
    QThreadPool.globalInstance().waitForDone(10000)
    assert frames, "no frames reached the GUI"
    assert frames[-1].result.faces, "the face in the test image was not detected"
    assert not worker.isRunning()
