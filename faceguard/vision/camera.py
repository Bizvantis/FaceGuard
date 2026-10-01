"""Frame sources: webcams (threaded, always-latest-frame), video files and still images."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


class CameraError(RuntimeError):
    pass


class FrameSource:
    """Base class. ``read`` returns the newest frame not returned before (or None on timeout)."""

    name = "source"
    is_live = False

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._count = 0
        self._returned = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.resolution: tuple[int, int] = (0, 0)

    # -- producer side -------------------------------------------------
    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._stamp = time.monotonic()
            self._count += 1
            self.resolution = (frame.shape[1], frame.shape[0])
            self._cond.notify_all()

    def _fail(self, message: str) -> None:
        log.warning("%s: %s", self.name, message)
        with self._cond:
            self.error = message
            self._running = False
            self._cond.notify_all()

    def _loop(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- consumer side -------------------------------------------------
    def start(self) -> "FrameSource":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-reader", daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self._running

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray | None, float]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._count == self._returned and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, 0.0
                self._cond.wait(remaining)
            if self._count == self._returned:
                return None, 0.0
            self._returned = self._count
            return self._frame, self._stamp

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        self._thread = None

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _backends() -> list[tuple[int, str]]:
    if sys.platform == "win32":
        return [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "auto")]
    return [(cv2.CAP_ANY, "auto")]


def open_capture(index: int, width: int, height: int, fps: int = 30) -> tuple[cv2.VideoCapture, str]:
    """Open a webcam, trying each Windows backend until one delivers frames."""
    tried: list[str] = []
    for api, api_name in _backends():
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            tried.append(f"{api_name}: could not open")
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return cap, api_name
            time.sleep(0.05)
        cap.release()
        tried.append(f"{api_name}: opened but returned no frames")
    raise CameraError(
        f"Camera {index} is not available ({'; '.join(tried)}). It may be in use by another app, "
        "or blocked in Windows Settings > Privacy & security > Camera (allow desktop apps)."
    )


class CameraSource(FrameSource):
    is_live = True

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        super().__init__()
        self.index = index
        self.name = f"camera{index}"
        self._size = (width, height)
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self.backend = ""

    def start(self) -> "CameraSource":
        self._cap, self.backend = open_capture(self.index, *self._size, self._fps)
        log.info("Opened camera %d via %s", self.index, self.backend)
        return super().start()  # type: ignore[return-value]

    def _loop(self) -> None:
        cap = self._cap
        failures = 0
        try:
            while self._running and cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 60:
                        self._fail("Camera stopped delivering frames (disconnected or taken by another app).")
                        break
                    time.sleep(0.03)
                    continue
                failures = 0
                self._publish(frame)
        finally:
            if cap is not None:
                cap.release()

    def stop(self) -> None:
        super().stop()
        self._cap = None


class VideoFileSource(FrameSource):
    """Plays a video file in real time, looping forever (demo / testing without a webcam)."""

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.name = self.path.name
        self.loop = loop

    def _loop(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            self._fail(f"Cannot open video {self.path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        period = 1.0 / max(1.0, min(fps, 60.0))
        try:
            next_t = time.monotonic()
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if not self.loop:
                        self._fail("End of video")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._publish(frame)
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
        finally:
            cap.release()


def load_image(path: str | Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII Windows paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


class ImageSource(FrameSource):
    """Cycles through still images at a fixed frame rate (each shown for ``hold`` seconds)."""

    def __init__(self, path: str | Path, fps: float = 15.0, hold: float = 3.0) -> None:
        super().__init__()
        p = Path(path)
        self.files = list_images(p) if p.is_dir() else [p]
        self.name = p.name
        self.fps = fps
        self.hold = hold

    def _loop(self) -> None:
        images = [img for img in (load_image(f) for f in self.files) if img is not None]
        if not images:
            self._fail("No readable images")
            return
        period = 1.0 / self.fps
        i, shown = 0, 0.0
        while self._running:
            # Publish a fresh copy each time so consumers see a "new" frame, like a real camera.
            self._publish(images[i].copy())
            time.sleep(period)
            shown += period
            if shown >= self.hold:
                i, shown = (i + 1) % len(images), 0.0


def open_source(spec: str | int, width: int = 1280, height: int = 720) -> FrameSource:
    """Camera index (``0``), video file, image file or folder of images."""
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return CameraSource(int(spec), width, height)
    p = Path(spec)
    if not p.exists():
        raise CameraError(f"Source not found: {spec}")
    if p.is_dir() or p.suffix.lower() in IMAGE_EXTS:
        return ImageSource(p)
    return VideoFileSource(p)


def probe_cameras(max_index: int = 4) -> list[tuple[int, str]]:
    """Return ``(index, "WxH via backend")`` for each camera that opens. Briefly activates each camera."""
    found = []
    for idx in range(max_index):
        try:
            cap, backend = open_capture(idx, 640, 480)
        except CameraError:
            continue
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found.append((idx, f"{w}x{h} via {backend}"))
    return found
