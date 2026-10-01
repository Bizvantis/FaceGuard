"""System-level evaluation of the presence guard on simulated webcam streams built from LFW photos.

Frames are 640x480; the face photo (~93 px wide face) is pasted with per-frame jitter (translation,
gain, sensor noise). The production engine (detection threshold 0.8, top-3 matching, EMA smoothing,
quality gate) and the production PresenceGuard run at the background rate of 4 frames per second.

A  owner at the desk: 20 owners x 10 simulated minutes, cycling through their other photos
   -> false intruder alerts, and false walk-away locks assuming no keyboard/mouse use at all.
B  owner leaves (5 s empty), a stranger sits down -> intruder lock latency.
C  a stranger replaces the owner with no gap -> intruder lock latency.
D  owner present, an unknown person appears behind them -> shoulder-surfer alert latency.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import lfw_index, make_engine, save_json

from faceguard.recognition.engine import EngineOptions
from faceguard.recognition.gallery import Gallery, PersonInfo
from faceguard.security.guard import GuardConfig, PresenceGuard, guard_faces
from faceguard.vision.camera import load_image

DT = 0.25
OPTS = EngineOptions(embed_interval=0.2)
_ENG = None


def _init():
    global _ENG
    cv2.setNumThreads(2)
    _ENG = make_engine(0.8)


class Stream:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def frame(self, photos, scales=(1.0,), offsets=((0, 0),)):
        """Compose one noisy 640x480 frame containing the given photos (or none)."""
        r = self.rng
        canvas = np.full((480, 640, 3), 70, np.float32)
        for img, s, (ox, oy) in zip(photos, scales, offsets):
            im = cv2.resize(img, None, fx=s, fy=s) if s != 1.0 else img
            h, w = im.shape[:2]
            x = int(320 - w / 2 + ox + r.integers(-3, 4))
            y = int(240 - h / 2 + oy + r.integers(-3, 4))
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(640, x + w), min(480, y + h)
            canvas[y0:y1, x0:x1] = im[y0 - y:y1 - y, x0 - x:x1 - x]
        canvas = canvas * r.uniform(0.9, 1.1) + r.normal(0, 2.0, canvas.shape)
        return np.clip(canvas, 0, 255).astype(np.uint8)


def enroll(paths):
    embs = []
    for p in paths:
        img = load_image(p)
        faces = _ENG.detector.detect(img)
        if faces:
            embs.append(_ENG.embed_face(img, faces[0])[1])
    return Gallery([PersonInfo("owner", "Owner", is_owner=True)], [np.stack(embs)], version=1)


def owner_session(args):
    """Scenario A: returns per-frame statuses and guard events for one owner."""
    seed, gal_paths, stream_paths, minutes = args
    _ENG.reset()
    _ENG.set_gallery(enroll(gal_paths))
    rng = random.Random(seed)
    st = Stream(seed)
    photos = [load_image(p) for p in stream_paths]
    typing = PresenceGuard(GuardConfig())
    idle = PresenceGuard(GuardConfig())
    typing.start(0.0)
    idle.start(0.0)
    status = defaultdict(int)
    events = defaultdict(int)
    t, current, until = 0.0, None, 0.0
    longest_gap, gap_start = 0.0, None
    while t < minutes * 60:
        if t >= until:
            current = rng.choice(photos)
            until = t + rng.uniform(1.0, 6.0)
        res = _ENG.process(st.frame([current]), t, OPTS)
        f = res.primary
        status[f.match.status if f else "no_face"] += 1
        faces = guard_faces(res)
        authorized = any(g.authorized for g in faces)
        if authorized:
            if gap_start is not None:
                longest_gap = max(longest_gap, t - gap_start)
                gap_start = None
        elif gap_start is None:
            gap_start = t
        for guard, idle_s, tag in ((typing, 0.0, "typing"), (idle, 1e9, "idle")):
            for a in guard.update(t, faces, idle_s, True):
                events[f"{tag}/{a.kind}/{a.reason}"] += 1
                if a.kind == "lock":
                    guard.on_session_locked(t)
                    guard.on_session_unlocked(t + 1.0)
        t += DT
    frames = sum(status.values())
    return {"frames": frames, "status": dict(status), "events": dict(events), "longest_unrecognised_s": longest_gap}


def intruder_trial(args):
    """Scenarios B/C/D: time (s) from the stranger's first frame to the first relevant guard action."""
    seed, gal_paths, owner_paths, stranger_path, gap, kind = args
    _ENG.reset()
    _ENG.set_gallery(enroll(gal_paths))
    rng = random.Random(seed)
    st = Stream(seed)
    owner_imgs = [load_image(p) for p in owner_paths]
    stranger = load_image(stranger_path)
    guard = PresenceGuard(GuardConfig())
    guard.start(0.0)
    t, arrive = 0.0, 20.0 + gap
    owner_img = rng.choice(owner_imgs)
    while t < arrive + 20.0:
        if t < 20.0:
            photos = [owner_img]
        elif t < arrive:
            photos = []
        else:
            photos = [stranger] if kind != "shoulder" else [owner_img, stranger]
        if kind == "shoulder" and t >= 20.0:
            frame = st.frame(photos, scales=(1.0, 0.7), offsets=((-60, 30), (150, -60))) if t >= arrive else st.frame([owner_img], offsets=((-60, 30),))
        else:
            frame = st.frame(photos)
        res = _ENG.process(frame, t, OPTS)
        for a in guard.update(t, guard_faces(res), 0.0, True):
            wanted = "lock" if kind != "shoulder" else "notify"
            if a.kind == wanted and a.reason in ("intruder", "shoulder") and t >= arrive:
                return t - arrive
        t += DT
    return None


OWNERS_A, MINUTES_A, SEED_A = 40, 15.0, 70  # scenario A: 40 owners x 15 min = 10 simulated hours
SEED_BCD = 7  # scenarios B-D (also replayed by exp_guard_diag.py)


def owner_tasks(owners, by):
    rng = random.Random(SEED_A)
    tasks = []
    for k, name in enumerate(rng.sample(owners, OWNERS_A)):
        paths = rng.sample(by[name], len(by[name]))
        tasks.append((k, paths[:10], paths[10:], MINUTES_A))
    return tasks


def main():
    import sys
    only_a = "--only-a" in sys.argv  # keep the B-D results of a previous run
    rng = random.Random(SEED_BCD)
    index = lfw_index()
    by = defaultdict(list)
    for name, _, path in index:
        by[name].append(path)
    owners = sorted(n for n, v in by.items() if len(v) >= 20)
    everyone = sorted(by)
    out = {"dt": DT, "config": GuardConfig().__dict__, "seed_a": SEED_A, "seed_bcd": SEED_BCD}
    # advance the B-D generator exactly as the original 20-owner run did, so trials B-D are unchanged
    for name in rng.sample(owners, 20):
        rng.sample(by[name], len(by[name]))

    tasks = owner_tasks(owners, by)
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init) as pool:
        sessions = list(pool.map(owner_session, tasks))
    print(f"A done in {time.time() - t0:.0f}s")
    frames = sum(s["frames"] for s in sessions)
    status = defaultdict(int)
    events = defaultdict(int)
    for s in sessions:
        for k, v in s["status"].items():
            status[k] += v
        for k, v in s["events"].items():
            events[k] += v
    hours = frames * DT / 3600
    out["owner_at_desk"] = {
        "owners": len(sessions), "simulated_hours": hours, "frames": frames,
        "status_fraction": {k: v / frames for k, v in status.items()}, "events": dict(events),
        "false_intruder_locks_per_hour": events.get("typing/lock/intruder", 0) / hours,
        "false_walkaway_locks_per_hour_no_typing": events.get("idle/lock/absence", 0) / hours,
        "longest_unrecognised_s": [s["longest_unrecognised_s"] for s in sessions],
    }
    print(out["owner_at_desk"])
    if only_a:
        from common import load_json
        old = load_json("guard")
        for kind in ("intruder", "replace", "shoulder"):
            out[f"latency_{kind}"] = old[f"latency_{kind}"]
        save_json("guard", out)
        return

    trials = []
    for k in range(300):
        name = rng.choice(owners)
        paths = rng.sample(by[name], len(by[name]))
        stranger = rng.choice(by[rng.choice([n for n in everyone if n != name])])
        trials.append((1000 + k, paths[:10], paths[10:], stranger, 5.0, "intruder"))
        trials.append((2000 + k, paths[:10], paths[10:], stranger, 0.0, "replace"))
        if k < 150:
            trials.append((3000 + k, paths[:10], paths[10:], stranger, 0.0, "shoulder"))
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init) as pool:
        lat = list(pool.map(intruder_trial, trials, chunksize=4))
    print(f"B/C/D done in {time.time() - t0:.0f}s")
    for kind in ("intruder", "replace", "shoulder"):
        vals = [v for tr, v in zip(trials, lat) if tr[5] == kind]
        ok = np.array([v for v in vals if v is not None])
        out[f"latency_{kind}"] = {"trials": len(vals), "detected_within_20s": len(ok) / len(vals),
                                  "mean_s": float(ok.mean()) if len(ok) else None, "median_s": float(np.median(ok)) if len(ok) else None,
                                  "p95_s": float(np.percentile(ok, 95)) if len(ok) else None, "max_s": float(ok.max()) if len(ok) else None}
        print(kind, out[f"latency_{kind}"])
    save_json("guard", out)


if __name__ == "__main__":
    main()
