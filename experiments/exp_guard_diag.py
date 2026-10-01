"""Failure analysis for exp_guard.py: why some strangers were not acted on, and where the owner-only
shoulder-surfer alerts came from.

Rebuilds exactly the same owners, streams and trials as exp_guard.py (same seeds) and runs the production
engine on a few frames of each static composition, recording how the stranger's face is classified.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import exp_guard as eg
from common import lfw_index, save_json

from faceguard.recognition.engine import EngineOptions
from faceguard.security.guard import GuardConfig, guard_faces
from faceguard.vision.camera import load_image

CFG = GuardConfig()


def settle(stream, photos, **kw):
    """Process four frames of a static scene (enough for two embeddings and EMA) and return the last result."""
    eg._ENG.reset()
    res = None
    for i in range(4):
        res = eg._ENG.process(stream.frame(photos, **kw), i * eg.DT, eg.OPTS)
    return res


def classify(face, gf, min_px):
    if face is None:
        return "no face detected"
    if gf.face_px < min_px:
        return "face too small"
    if not gf.quality_ok:
        return "quality gate"
    if gf.authorized:
        return "accepted as the owner"
    if not gf.stranger:
        return "uncertain band"
    return "unknown (acted on)"


def main():
    eg._init()
    rng = random.Random(7)
    by = defaultdict(list)
    for name, _, path in lfw_index():
        by[name].append(path)
    owners = sorted(n for n, v in by.items() if len(v) >= 20)
    everyone = sorted(by)

    # advance the B-D generator exactly as exp_guard.py does
    for name in rng.sample(owners, 20):
        rng.sample(by[name], len(by[name]))

    # scenario A (same owners and streams as exp_guard.py): which streamed photos show a second, unknown face?
    second = Counter()
    per_owner = []
    for k, gal_paths, stream_paths, _ in eg.owner_tasks(owners, by):
        name = str(gal_paths[0]).replace("\\", "/").split("/")[-2]
        paths = list(gal_paths) + list(stream_paths)
        eg._ENG.set_gallery(eg.enroll(paths[:10]))
        st = eg.Stream(k)
        n_second = 0
        for p in paths[10:]:
            res = settle(st, [load_image(p)])
            others = [g for g in guard_faces(res) if g.stranger and g.face_px >= CFG.min_watcher_face_px]
            second["photos"] += 1
            second["faces>=2"] += len(res.faces) >= 2
            if others:
                n_second += 1
                second["unknown second face"] += 1
        per_owner.append({"owner": name, "stream_photos": len(paths) - 10, "photos_with_unknown_second_face": n_second})
    print("scenario A:", dict(second))

    # scenarios B/C/D: static classification of the stranger in each trial
    out = {"owner_stream": {"totals": dict(second), "per_owner": per_owner}}
    reasons = {"intruder": Counter(), "shoulder": Counter()}
    for k in range(300):
        name = rng.choice(owners)
        paths = rng.sample(by[name], len(by[name]))
        stranger = rng.choice(by[rng.choice([n for n in everyone if n != name])])
        eg._ENG.set_gallery(eg.enroll(paths[:10]))
        simg = load_image(stranger)
        res = settle(eg.Stream(1000 + k), [simg])
        f = res.primary
        gf = next((g for g, ff in zip(guard_faces(res), res.faces) if ff is f), None)
        reasons["intruder"][classify(f, gf, CFG.min_intruder_face_px)] += 1
        if k < 150:
            owner_img = load_image(paths[10])
            res = settle(eg.Stream(3000 + k), [owner_img, simg], scales=(1.0, 0.7), offsets=((-60, 30), (150, -60)))
            # the stranger is pasted right of centre
            cand = [(g, ff) for g, ff in zip(guard_faces(res), res.faces) if ff.bbox[0] + ff.bbox[2] / 2 > 320]
            g, ff = max(cand, key=lambda c: c[0].face_px) if cand else (None, None)
            reasons["shoulder"][classify(ff, g, CFG.min_watcher_face_px)] += 1
    for kind, c in reasons.items():
        print(kind, dict(c))
    out["stranger_classification"] = {k: dict(v) for k, v in reasons.items()}
    save_json("guard_diag", out)


if __name__ == "__main__":
    main()
