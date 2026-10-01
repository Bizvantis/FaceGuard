"""Application entry point: single instance, first-run model download, window / tray start-up."""

from __future__ import annotations

import getpass
import logging
import sys

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from faceguard import APP_NAME, models
from faceguard.config import Settings
from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.main_window import MainWindow
from faceguard.gui.widgets import run_job
from faceguard.gui.worker import VideoWorker
from faceguard.logs import setup_logging
from faceguard.recognition.database import FaceDatabase
from faceguard.security.intruders import IntruderLog

log = logging.getLogger(__name__)


def _server_name() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover
        user = "user"
    return f"{APP_NAME}-{user}"


def _already_running() -> bool:
    """If another instance runs for this user, ask it to show its window and return True."""
    sock = QLocalSocket()
    sock.connectToServer(_server_name())
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return True
    return False


def _ensure_models() -> bool:
    """Download missing models with a progress dialog. Returns False if the user cancels or it fails."""
    missing = [k for k in models.MODELS if not models.is_present(k)]
    if not missing:
        return True
    size = sum(models.MODELS[k].size for k in missing) / 1e6
    dlg = QProgressDialog(f"Downloading the neural-network models ({size:.0f} MB, one time only)...", "Cancel", 0, 0)
    dlg.setWindowTitle(APP_NAME)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.show()
    loop = QEventLoop()
    outcome: dict = {}
    run_job(models.ensure_models, list(models.MODELS),
            on_done=lambda r: (outcome.update(ok=True), loop.quit()),
            on_error=lambda e: (outcome.update(error=e), loop.quit()))
    dlg.canceled.connect(loop.quit)
    loop.exec()
    dlg.close()
    if outcome.get("ok"):
        return True
    QMessageBox.critical(None, APP_NAME, "The face-recognition models could not be downloaded.\n\n"
                         f"{outcome.get('error', 'Cancelled')}\n\nCheck your internet connection, or run:\n"
                         "  python -m faceguard download-models")
    return all(models.is_present(k) for k in models.REQUIRED)


def run(tray_only: bool = False, source: str | None = None) -> int:
    setup_logging(console=False)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setWindowIcon(icons.app_icon())
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    app.setQuitOnLastWindowClosed(False)

    if _already_running():
        log.info("Another instance is running; asked it to show its window")
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(_server_name())
    server.listen(_server_name())

    if not _ensure_models():
        return 1
    settings = Settings.load()
    db = FaceDatabase.open(encrypt=settings.encrypt_database)
    intruders = IntruderLog()
    removed = intruders.cleanup(settings.intruder_retention_days)
    if removed:
        log.info("Removed %d old snapshots", removed)

    worker = VideoWorker(settings, db, source)
    ctx = AppContext(settings, db, worker, intruders)
    win = MainWindow(ctx)

    def on_connection():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.disconnectFromServer()
        win.show_normal()

    server.newConnection.connect(on_connection)
    worker.start()

    owner = db.owner()
    has_owner = owner is not None and owner.count() > 0
    if tray_only and has_owner and not settings.guard_enabled:
        settings.guard_enabled = True
        settings.save()
    if settings.guard_enabled and has_owner:
        worker.guard_command("start")
    elif settings.guard_enabled:
        settings.guard_enabled = False
    win.by_key["guard"].toggle.setChecked(settings.guard_enabled)
    win.sync_tray_guard()

    if (tray_only or settings.start_minimized) and win.tray is not None:
        if settings.guard_enabled:
            win.tray.showMessage(APP_NAME, "FaceGuard is protecting your PC.", icons.app_icon(), 3000)
    else:
        win.show()
    if db.load_warning:
        QMessageBox.warning(win, APP_NAME, db.load_warning)
    code = app.exec()
    worker.stop()
    return code


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
