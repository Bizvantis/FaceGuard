"""Runtime benchmark: per-stage latency vs. resolution, model load time, memory, CPU load of the camera thread."""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

from common import ROOT, make_engine, save_json

from faceguard import models
from faceguard.recognition.engine import EngineOptions
from faceguard.vision.camera import load_image
from faceguard.vision.mesh import MeshAnalyzer

N_WARM, N_RUNS = 20, 200
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


def working_set_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 2**20


def scene(width: int, height: int) -> np.ndarray:
    """A test frame with a face occupying ~22% of the frame width (typical laptop-webcam distance)."""
    face = load_image(ROOT / "tests" / "data" / "astronaut.jpg")  # 320x320 crop, face ~90 px wide
    scale = 0.22 * width / 90.0
    face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 96, np.uint8)
    fh, fw = face.shape[:2]
    y0, x0 = max(0, (height - fh) // 2), max(0, (width - fw) // 2)
    crop = face[: min(fh, height), : min(fw, width)]
    canvas[y0: y0 + crop.shape[0], x0: x0 + crop.shape[1]] = crop
    rng = np.random.default_rng(0)
    return np.clip(canvas.astype(np.int16) + rng.normal(0, 2, canvas.shape).astype(np.int16), 0, 255).astype(np.uint8)


def timeit(fn, n=N_RUNS, warm=N_WARM) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"mean": statistics.fmean(ts), "std": statistics.pstdev(ts), "p50": ts[len(ts) // 2], "p95": ts[int(0.95 * len(ts))]}


def cpu_load(view_active: bool, seconds: float = 20.0) -> dict:
    """Run the real camera thread on a 30 fps synthetic stream and measure process CPU time."""
    from PySide6.QtCore import QCoreApplication, QTimer

    from faceguard.config import Settings
    from faceguard.gui.worker import VideoWorker
    from faceguard.recognition.database import FaceDatabase, Sample
    from faceguard.vision.camera import ImageSource

    os.environ["FACEGUARD_HOME"] = str(ROOT / "experiments" / "data" / "runtime_home")
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    frame_path = ROOT / "experiments" / "data" / "runtime_frame.png"
    cv2.imwrite(str(frame_path), scene(1280, 720))
    db = FaceDatabase(ROOT / "experiments" / "data" / "runtime_home" / "faces.db")
    if not db.persons:
        eng = make_engine(0.8)
        img = scene(1280, 720)
        f = eng.detector.detect(img)[0]
        p = db.add_person("Owner")
        db.add_samples(p.id, [Sample(eng.embed_face(img, f)[1])])
    import faceguard.security.winsys as ws

    ws.lock_workstation = lambda: True
    import faceguard.vision.camera as cam

    orig = cam.open_source
    cam.open_source = lambda spec, w, h: ImageSource(spec, fps=30, hold=1e9)
    import faceguard.gui.worker as wk

    wk.open_source = cam.open_source
    worker = VideoWorker(Settings(), db, str(frame_path))
    frames = [0]
    worker.frame_ready.connect(lambda p: (frames.__setitem__(0, frames[0] + 1), worker.frame_consumed()))
    worker.start()
    worker.set_view_active(view_active)
    if not view_active:
        worker.guard_command("start")
    time.sleep(4.0)  # warm up (model load, camera start)
    app.processEvents()
    c0, w0, f0 = time.process_time(), time.perf_counter(), frames[0]
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)
    cpu = time.process_time() - c0
    wall = time.perf_counter() - w0
    shown = frames[0] - f0
    worker.stop()
    cam.open_source = orig
    wk.open_source = orig
    return {"cpu_percent_of_one_core": 100.0 * cpu / wall, "cpu_percent_of_machine": 100.0 * cpu / wall / os.cpu_count(),
            "frames_displayed_per_s": shown / wall, "seconds": wall}


def main() -> None:
    out: dict = {
        "platform": {
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
        }
    }
    try:
        import mediapipe

        out["platform"]["mediapipe"] = mediapipe.__version__
    except ImportError:
        pass
    mem0 = working_set_mb()
    t0 = time.perf_counter()
    engine = make_engine(0.8)
    t1 = time.perf_counter()
    mesh = MeshAnalyzer(models.MODELS["landmarker"].path, max_faces=4)
    t2 = time.perf_counter()
    out["load"] = {"detector_and_recognizer_ms": (t1 - t0) * 1e3, "mesh_ms": (t2 - t1) * 1e3,
                   "working_set_before_mb": mem0, "working_set_after_mb": working_set_mb()}
    out["stages"] = {}
    for w, h in RESOLUTIONS:
        img = scene(w, h)
        faces = engine.detector.detect(img)
        assert faces, f"no face at {w}x{h}"
        face = faces[0]
        aligned = engine.embedder.align(img, face)
        ts = [0.0]

        def mesh_step():
            ts[0] += 1 / 30
            mesh.process(img, ts[0])

        def pipeline(opts):
            def run():
                ts[0] += 1 / 30
                engine.process(img, ts[0], opts)
            return run

        engine.reset()
        r = {
            "face_px": float(face.width),
            "detect": timeit(lambda: engine.detector.detect(img)),
            "align": timeit(lambda: engine.embedder.align(img, face)),
            "embed": timeit(lambda: engine.embedder.embed_aligned(aligned, tta=False)),
            "embed_tta": timeit(lambda: engine.embedder.embed_aligned(aligned, tta=True)),
            "mesh": timeit(mesh_step),
            "pipeline_embed_every_frame": timeit(pipeline(EngineOptions(embed_interval=0.0))),
            "pipeline_default": timeit(pipeline(EngineOptions())),
            "pipeline_with_mesh": timeit(pipeline(EngineOptions(mesh=True))),
        }
        out["stages"][f"{w}x{h}"] = r
        print(f"{w}x{h}: detect {r['detect']['mean']:.1f} ms, embed {r['embed']['mean']:.1f}, tta {r['embed_tta']['mean']:.1f}, "
              f"mesh {r['mesh']['mean']:.1f}, pipeline {r['pipeline_default']['mean']:.1f} / every-frame "
              f"{r['pipeline_embed_every_frame']['mean']:.1f} / +mesh {r['pipeline_with_mesh']['mean']:.1f}")
    out["memory_after_benchmark_mb"] = working_set_mb()
    mesh.close()
    print("CPU load: live view ...")
    out["cpu_live_view"] = cpu_load(view_active=True)
    print("CPU load: background guard ...")
    out["cpu_guard_background"] = cpu_load(view_active=False)
    print(out["cpu_live_view"], out["cpu_guard_background"], sep="\n")
    save_json("runtime", out)


if __name__ == "__main__":
    main()
