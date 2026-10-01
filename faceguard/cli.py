"""Command-line interface.  Run ``python -m faceguard --help``.

With no arguments the desktop app starts.  Every feature is also scriptable,
e.g. ``python -m faceguard verify --owner && echo unlocked`` gates a script on
a live face check (exit code 0 = verified).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from faceguard import APP_NAME, __version__, models, paths
from faceguard.config import Settings
from faceguard.logs import setup_logging

log = logging.getLogger("faceguard")


# ---------------------------------------------------------------- helpers
def _db(settings: Settings):
    from faceguard.recognition.database import FaceDatabase

    db = FaceDatabase.open(encrypt=settings.encrypt_database)
    if db.load_warning:
        print("WARNING:", db.load_warning, file=sys.stderr)
    return db


def _engine(settings: Settings, db=None):
    from faceguard.recognition.engine import FaceEngine

    try:
        engine = FaceEngine.create(settings)
    except models.ModelError as exc:
        raise SystemExit(str(exc))
    if db is not None:
        engine.set_gallery(db.snapshot())
    return engine


def _source(args, settings: Settings):
    from faceguard.vision.camera import CameraError, open_source

    spec = getattr(args, "source", None)
    spec = spec if spec is not None else settings.camera_index
    try:
        return open_source(spec, settings.camera_width, settings.camera_height).start()
    except CameraError as exc:
        raise SystemExit(str(exc))


def _window_loop(source, title: str, step) -> None:
    """Run ``step(frame, t) -> (image, done)`` on each frame until done or Esc/Q."""
    import cv2

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, t = source.read(1.0)
            if frame is None:
                if source.error:
                    raise SystemExit(source.error)
                continue
            image, done = step(frame, t)
            cv2.imshow(title, image)
            key = cv2.waitKey(1) & 0xFF
            if done or key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        source.stop()
        cv2.destroyWindow(title)


# ---------------------------------------------------------------- commands
def cmd_gui(args, settings: Settings) -> int:
    from faceguard.gui.app import run

    return run(tray_only=args.command == "tray", source=getattr(args, "source", None))


def cmd_download_models(args, settings: Settings) -> int:
    def progress(key, done, total):
        print(f"\r  {models.MODELS[key].filename}: {100 * done / max(1, total):5.1f}%", end="", flush=True)

    for key, spec in models.MODELS.items():
        if models.is_present(key, verify=True):
            print(f"OK   {spec.filename} ({spec.title})")
            continue
        print(f"Downloading {spec.filename} from {spec.source} ({spec.size / 1e6:.1f} MB)")
        models.ensure_models([key], progress)
        print()
    return 0


def cmd_doctor(args, settings: Settings) -> int:
    import platform

    import cv2
    import numpy as np

    from faceguard.security import dpapi, winsys
    from faceguard.vision.mesh import mediapipe_available

    ok = True
    print(f"{APP_NAME} {__version__} on Python {platform.python_version()} ({platform.platform()})")
    print(f"OpenCV {cv2.__version__}, NumPy {np.__version__}")
    for key, spec in models.MODELS.items():
        present = models.is_present(key, verify=True)
        ok &= present or key == "landmarker"
        print(f"  model {spec.filename:40s} {'OK' if present else 'MISSING (run: python -m faceguard download-models)'}")
    print(f"  MediaPipe (blink/smile liveness): {'available' if mediapipe_available() else 'not installed'}")
    try:
        import PySide6

        print(f"  PySide6 {PySide6.__version__} (desktop app): OK")
    except ImportError:
        ok = False
        print("  PySide6: MISSING - the desktop app will not start")
    print(f"  DPAPI encryption: {'available' if dpapi.available() else 'not available (non-Windows)'}")
    print(f"  Session lock state: {winsys.is_session_locked()}  idle: {winsys.idle_seconds():.0f}s")
    print(f"  Data folder: {paths.data_dir()}")
    if args.camera:
        from faceguard.vision.camera import probe_cameras

        cams = probe_cameras()
        print("  Cameras:", ", ".join(f"#{i} ({d})" for i, d in cams) or "none found")
        ok &= bool(cams)
    else:
        print("  Cameras: not checked (add --camera to test; this briefly turns the webcam on)")
    if models.is_present("detector") and models.is_present("recognizer"):
        engine = _engine(settings)
        frame = np.zeros((480, 640, 3), np.uint8)
        t0 = time.perf_counter()
        for _ in range(10):
            engine.process(frame)
        print(f"  Detector speed on this CPU: {(time.perf_counter() - t0) * 100:.1f} ms/frame (640x480)")
    print("All good!" if ok else "Some checks failed (see above).")
    return 0 if ok else 1


def cmd_enroll(args, settings: Settings) -> int:
    from faceguard.recognition.enrollment import EnrollmentSession, find_duplicate, samples_from_photos
    from faceguard.vision import drawing

    db = _db(settings)
    engine = _engine(settings, db)
    person = db.find(args.name)
    if args.images:
        files: list[Path] = []
        from faceguard.vision.camera import list_images

        for item in args.images:
            p = Path(item)
            files.extend(list_images(p) if p.is_dir() else [p])
        report = samples_from_photos(engine, files)
        for name, reason in report.skipped:
            print(f"  skipped {name}: {reason}")
        if not report.samples:
            print("No usable faces found.")
            return 1
        samples, thumbs = report.samples, report.thumbnails
        print(f"Using {len(samples)} photo(s).")
    else:
        session = EnrollmentSession(args.name)
        source = _source(args, settings)

        def step(frame, t):
            result = engine.process(frame, t)
            state = session.update(frame, result, engine, t)
            img = drawing.draw_results(frame, result, settings.mirror_preview)
            drawing.panel(img, [state.instruction, state.hint or " ", f"{int(state.progress * 100)}%   (S = skip pose, Esc = cancel)"])
            drawing.progress_bar(img, state.progress)
            return img, state.phase != "running"

        import cv2

        def step_keys(frame, t):
            img, done = step(frame, t)
            if cv2.waitKey(1) & 0xFF == ord("s"):
                session.skip_current(t)
            return img, done

        _window_loop(source, f"{APP_NAME} - enroll {args.name}", step_keys)
        if session.phase != "done":
            print("Enrollment cancelled.")
            return 1
        res = session.finish()
        samples, thumbs = res.samples, res.thumbnails
        print(f"Captured {len(samples)} samples (consistency {res.consistency:.2f}); skipped poses: {res.skipped or 'none'}")
    import numpy as np

    from faceguard.vision.embedder import normalize

    centroid = normalize(np.mean([s.embedding for s in samples], axis=0))
    dup = find_duplicate(db.snapshot(), engine.matcher, centroid, exclude_id=person.id if person else None)
    if dup and not args.force:
        print(f"This face already matches '{dup.person.name}' (similarity {dup.similarity:.2f}). Use --force to enroll anyway.")
        return 1
    if person is None:
        person = db.add_person(args.name, is_owner=True if args.owner else None)
    elif args.owner:
        db.set_owner(person.id)
    db.add_samples(person.id, samples, thumbs)
    person = db.get(person.id)
    print(f"Enrolled {person.name}: {person.count()} samples{' (owner)' if person.is_owner else ''}.")
    return 0


def cmd_recognize(args, settings: Settings) -> int:
    import cv2

    from faceguard.vision import drawing
    from faceguard.vision.camera import load_image

    db = _db(settings)
    engine = _engine(settings, db)
    if args.image:
        for path in args.image:
            img = load_image(path)
            if img is None:
                print(f"{path}: cannot read image")
                continue
            found = engine.analyze_image(img)
            print(f"{path}: {len(found)} face(s)")
            for f in found:
                m = f.match
                print(f"  {m.label:20s} similarity {m.similarity:.3f}  quality {f.quality.score:.2f}  box {f.face.bbox.round(0).tolist()}")
            if args.save:
                out = img.copy()
                for f in found:
                    x, y, w, h = (int(v) for v in f.face.bbox)
                    color = drawing.GREEN if f.match.is_match else drawing.RED
                    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                    drawing.label(out, f"{f.match.label} {f.match.similarity:.2f}", (x, y - 4), color)
                target = Path(args.save)
                target = target / Path(path).name if target.is_dir() else target
                cv2.imencode(Path(target).suffix or ".jpg", out)[1].tofile(str(target))
                print(f"  saved {target}")
        return 0
    source = _source(args, settings)
    fps = [time.perf_counter()]

    def step(frame, t):
        from faceguard.recognition.engine import EngineOptions

        result = engine.process(frame, t, EngineOptions(mesh=args.mesh))
        img = drawing.draw_results(frame, result, settings.mirror_preview)
        now = time.perf_counter()
        rate = 1.0 / max(1e-6, now - fps[0])
        fps[0] = now
        drawing.panel(img, [f"{rate:4.1f} FPS  detect {result.timings['detect_ms']:.0f} ms  embed {result.timings['embed_ms']:.0f} ms",
                            f"{len(result.faces)} face(s), {result.gallery_size} enrolled   (Esc to quit)"])
        return img, False

    _window_loop(source, f"{APP_NAME} - recognize", step)
    return 0


def cmd_verify(args, settings: Settings) -> int:
    from faceguard.recognition.engine import EngineOptions
    from faceguard.security.liveness import LivenessConfig, VerificationTask
    from faceguard.vision import drawing

    db = _db(settings)
    engine = _engine(settings, db)
    expressions = settings.liveness_expressions and engine.ensure_mesh()
    cfg = LivenessConfig(num_challenges=settings.liveness_challenges, challenge_timeout=settings.liveness_timeout)
    task = VerificationTask(db.snapshot(), engine.matcher, cfg, expressions, require_owner=args.owner)
    options = EngineOptions(mesh=task.needs_mesh, embed_interval=0.3)
    source = _source(args, settings)
    done_at = [None]

    def step(frame, t):
        result = engine.process(frame, t, options)
        state = task.update(result)
        img = drawing.draw_results(frame, result, settings.mirror_preview, landmarks=False)
        lines = [f"Step {state.step}/{state.total}: {state.instruction}", state.hint or " "]
        if task.result is not None:
            r = task.result
            lines = [("VERIFIED: " + r.match.label) if r.verified else ("DENIED: " + (r.reason or "not verified"))]
            done_at[0] = done_at[0] or t
        drawing.panel(img, lines, scale=0.8)
        drawing.progress_bar(img, state.progress)
        return img, done_at[0] is not None and t - done_at[0] > 1.5

    _window_loop(source, f"{APP_NAME} - verify", step)
    r = task.result
    if r is None:
        print("Verification cancelled.")
        return 2
    if r.verified:
        print(f"VERIFIED {r.match.person.name} (similarity {r.match.similarity:.2f}, liveness {r.duration:.1f}s)")
        return 0
    print(f"NOT VERIFIED: {r.reason} (live={r.live}, best={r.match.label} {r.match.similarity:.2f})")
    return 1


def cmd_people(args, settings: Settings) -> int:
    db = _db(settings)
    if args.action == "list":
        if not db.persons:
            print("Nobody is enrolled yet. Try: python -m faceguard enroll \"Your Name\"")
        for p in db.persons:
            role = "owner" if p.is_owner else ("trusted" if p.trusted else "known")
            print(f"{p.name:24s} {role:8s} {p.count():3d} samples ({p.count('adaptive')} adaptive)  "
                  f"enrolled {time.strftime('%Y-%m-%d', time.localtime(p.created))}")
        return 0
    person = db.find(args.name or "")
    if person is None:
        print(f"No person named '{args.name}'.")
        return 1
    if args.action == "delete":
        db.delete(person.id)
    elif args.action == "rename":
        db.rename(person.id, args.new_name)
    elif args.action == "owner":
        db.set_owner(person.id)
    elif args.action == "trust":
        db.set_trusted(person.id, True)
    elif args.action == "untrust":
        db.set_trusted(person.id, False)
    elif args.action == "reset-adaptive":
        print(f"Removed {db.clear_adaptive(person.id)} adaptive samples.")
    print("Done.")
    return 0


def cmd_analyze(args, settings: Settings) -> int:
    from faceguard.recognition.analysis import analyze

    db = _db(settings)
    r = analyze(db.persons, settings.match_threshold, settings.top_k, "tsne" if args.tsne else "pca")
    print(f"{len(r.persons)} people, {len(r.labels)} samples  ({r.elapsed:.2f}s)")
    for s in r.persons:
        near = f", closest: {s.nearest_other} ({s.nearest_other_sim:.2f})" if s.nearest_other else ""
        print(f"  {s.name:24s} {s.samples:3d} samples  consistency {s.consistency:.2f} (worst {s.worst:.2f}){near}")
    for name, acc in r.methods.items():
        print(f"  {name:28s} {100 * acc:6.2f}%")
    if r.auc is not None:
        print(f"  ROC AUC {r.auc:.4f}   EER {100 * r.eer:.2f}% at {r.eer_threshold:.3f}")
    print(f"  Current threshold {r.current_threshold:.2f}; recommended {r.recommended_threshold:.2f}")
    for n in r.notes:
        print("  note:", n)
    return 0


def cmd_benchmark(args, settings: Settings) -> int:
    from faceguard.recognition import evaluation as ev

    engine = _engine(settings)
    threshold = args.threshold if args.threshold is not None else settings.match_threshold

    def progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"\r  embedding {done}/{total}", end="", flush=True)

    if args.lfw:
        print("LFW benchmark (scikit-learn downloads ~230 MB the first time)...")
        rep = ev.evaluate_lfw(engine, args.data_home, progress)
    elif args.olivetti:
        rep = ev.evaluate_olivetti(engine, threshold, args.data_home, progress)
    elif args.folder:
        rep = ev.evaluate_folder(engine, args.folder, threshold, progress)
    else:
        print("Choose one of --folder DIR, --olivetti or --lfw")
        return 2
    print()
    print(rep.summary())
    if args.out:
        print("Saved", ev.save_report(rep, args.out))
    return 0


def cmd_guard(args, settings: Settings) -> int:
    """Headless guard in the console (the desktop app's tray mode is the normal way to run it)."""
    from faceguard.recognition.engine import EngineOptions
    from faceguard.security.runner import GuardRunner

    db = _db(settings)
    engine = _engine(settings, db)
    runner = GuardRunner(settings, db, notify=lambda k, title, msg: print(f"[{time.strftime('%H:%M:%S')}] {title}: {msg}"))
    source = None
    runner.start()
    print("Guard running - Ctrl+C to stop.")
    period = 1.0 / max(0.5, settings.guard_fps)
    try:
        while True:
            t = time.monotonic()
            locked = runner.poll_session(t)
            if locked and not runner.guard.watching_locked(t):
                if source:
                    source.stop()
                    source = None
                time.sleep(1.0)
                continue
            if source is None:
                source = _source(args, settings)
                runner.guard.on_camera_restored(time.monotonic())
            frame, ts = source.read(2.0)
            if frame is None:
                if source.error:
                    runner.guard.on_camera_lost(t)
                    source.stop()
                    source = None
                    time.sleep(5)
                continue
            if engine.gallery.version != db.version:
                engine.set_gallery(db.snapshot())
            result = engine.process(frame, ts, EngineOptions(embed_interval=0.2))
            if locked:
                runner.process_locked(frame, result, ts)
            else:
                runner.process(frame, result, ts)
            print(f"\r{runner.status_text(t):60s}", end="", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if source:
            source.stop()
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faceguard", description=f"{APP_NAME} {__version__} - face recognition & presence security")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command")

    def with_source(sp):
        sp.add_argument("--source", help="camera index, video file, image or folder (default: camera from settings)")
        return sp

    with_source(sub.add_parser("gui", help="start the desktop app (default)"))
    with_source(sub.add_parser("tray", help="start in the system tray with the guard on (used at sign-in)"))
    sub.add_parser("download-models", help="download and verify the neural-network models")
    d = sub.add_parser("doctor", help="check the installation")
    d.add_argument("--camera", action="store_true", help="also test the webcam(s)")

    e = with_source(sub.add_parser("enroll", help="enroll a person from the webcam or from photos"))
    e.add_argument("name")
    e.add_argument("--images", nargs="+", help="photo files or folders instead of the webcam")
    e.add_argument("--owner", action="store_true", help="make this person the owner of the PC")
    e.add_argument("--force", action="store_true", help="enroll even if the face matches someone else")

    r = with_source(sub.add_parser("recognize", help="recognise faces in images or live video"))
    r.add_argument("--image", nargs="+", help="image file(s) to analyse instead of video")
    r.add_argument("--save", help="write annotated image(s) to this file or folder")
    r.add_argument("--mesh", action="store_true", help="also run the dense face mesh")

    v = with_source(sub.add_parser("verify", help="liveness + face verification; exit code 0 if verified"))
    v.add_argument("--owner", action="store_true", help="only the owner counts as verified")

    pe = sub.add_parser("people", help="manage enrolled people")
    pe.add_argument("action", choices=["list", "delete", "rename", "owner", "trust", "untrust", "reset-adaptive"])
    pe.add_argument("name", nargs="?")
    pe.add_argument("new_name", nargs="?")

    a = sub.add_parser("analyze", help="ML analysis of the enrolled gallery")
    a.add_argument("--tsne", action="store_true")

    b = sub.add_parser("benchmark", help="evaluate accuracy on a labelled dataset")
    b.add_argument("--folder", help="dataset folder: one sub-folder of images per person")
    b.add_argument("--olivetti", action="store_true", help="AT&T/Olivetti faces (small download)")
    b.add_argument("--lfw", action="store_true", help="Labeled Faces in the Wild, 10-fold protocol (~230 MB download)")
    b.add_argument("--threshold", type=float)
    b.add_argument("--data-home", help="where scikit-learn stores downloaded datasets")
    b.add_argument("--out", help="folder for a JSON/TXT report")

    with_source(sub.add_parser("guard", help="run the presence guard in this console"))
    return p


COMMANDS = {
    "gui": cmd_gui, "tray": cmd_gui, "download-models": cmd_download_models, "doctor": cmd_doctor,
    "enroll": cmd_enroll, "recognize": cmd_recognize, "verify": cmd_verify, "people": cmd_people,
    "analyze": cmd_analyze, "benchmark": cmd_benchmark, "guard": cmd_guard,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        args.command = "gui"
        args.source = None
    setup_logging(console=args.command not in ("gui", "tray"), level=logging.WARNING if args.command not in ("gui", "tray", "guard") else logging.INFO)
    settings = Settings.load()
    return int(COMMANDS[args.command](args, settings) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
