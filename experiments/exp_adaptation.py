"""Template adaptation on laptop-webcam images recorded over weeks (MPIIFaceGaze): benefit, drift and poisoning.

Uses the image sample of exp_mpii.py (seed 2026, up to 300 frames per participant spread over their recording days)
and the production update rule of faceguard/recognition/adaptive.py and FaceDatabase.add_adaptive_sample:
a frame of an enrolled person is appended to that person's template only if it passes the passive quality gate,
|yaw| <= 25 deg, the face is >= 80 px wide, its top-3 score reaches tau + 0.12 (0.57 at the balanced level), its
highest similarity to the stored samples is below 0.92 and the previous update is at least 10 minutes old.
Adaptive samples form a FIFO buffer of min(30, max(5, n_enrolled)) entries.

MPIIFaceGaze frames carry no clock time, so the main run allows at most one update per recording day (fewer than
the app's one per 10 minutes); a stress run drops the rate limit, so every eligible frame is added.

Protocol (temporal split, as exp_mpii.py B): each participant is enrolled with the 22 frames of their first recording
days. The other frames of the first half of their days are streamed in chronological order through the rule; the
frames of the second half are the probes, scored with the static enrollment template and with the adapted one.
Impostors: each participant's probes against the 14 other templates (leave-one-person-out).
Poisoning: every frame of the 14 other participants is offered to each template through the same rule, as if an
impostor sat at the PC on every recording day.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import exp_mpii
from common import make_engine, save_json

from faceguard.recognition.database import MAX_ADAPTIVE
from faceguard.vision import quality as fq
from faceguard.vision.camera import load_image
from faceguard.vision.embedder import normalize

TAU = {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52}
MARGIN, NOVELTY, K, ENROLL = 0.12, 0.92, 3, exp_mpii.ENROLL
_ENG = None


def _init():
    global _ENG
    import cv2
    cv2.setNumThreads(1)
    _ENG = make_engine()


def _embed(path):
    img = load_image(path)
    faces = _ENG.detector.detect(img)
    if not faces:
        return None
    h, w = img.shape[:2]
    face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
    _, emb, q = _ENG.embed_face(img, face, tta=True, thresholds=fq.PASSIVE_THRESHOLDS)
    eligible = bool(q.ok and abs(q.yaw) <= 25 and q.face_px >= 80)  # the conditions of AdaptiveLearner.observe
    return normalize(np.asarray(emb, np.float32).reshape(-1)), eligible


def top_k(gal, q):
    return float(np.sort(gal @ q)[::-1][:K].mean())


def adapt(enrolled, stream, tau, rate_limit=True):
    """Run the production update rule over (embedding, eligible, day) in time order. Returns the adaptive samples."""
    cap = min(MAX_ADAPTIVE, max(5, len(enrolled)))
    adaptive, last_day, added_scores = [], None, []
    for e, ok, d in stream:
        if not ok:
            continue
        gal = np.vstack([enrolled] + adaptive) if adaptive else enrolled
        s = top_k(gal, e)
        if s < tau + MARGIN or float(np.max(gal @ e)) >= NOVELTY:
            continue
        if rate_limit and last_day == d:
            continue
        adaptive.append(e[None])
        added_scores.append(s)
        if len(adaptive) > cap:
            adaptive.pop(0)
        last_day = d
    return adaptive, added_scores


def evaluate(E, ok, person, day, ndays, tau, rate_limit):
    people = sorted(set(person))
    enrolled, stream, probes = {}, {}, {}
    for p in people:
        idx = np.where(person == p)[0]
        order = idx[np.argsort(day[idx], kind="stable")]
        late = day[order] >= ndays[order] / 2.0  # second half of this person's recording days
        enrolled[p] = order[:ENROLL]
        stream[p] = order[ENROLL:][~late[ENROLL:]]  # chronological, first half of the days
        probes[p] = np.setdiff1d(order[late], enrolled[p])
    static = {p: E[enrolled[p]] for p in people}
    adapted, n_updates = {}, {}
    for p in people:
        a, _ = adapt(static[p], [(E[i], ok[i], day[i]) for i in stream[p]], tau, rate_limit)
        adapted[p] = np.vstack([static[p]] + a) if a else static[p]
        n_updates[p] = len(a)
    out = {"updates_per_person": n_updates, "updates_total": int(sum(n_updates.values())),
           "stream_frames": int(sum(len(v) for v in stream.values())), "probes": int(sum(len(v) for v in probes.values()))}
    for name, gals in (("static", static), ("adapted", adapted)):
        gen, imp = [], []
        for p in people:
            for q in probes[p]:
                gen.append(top_k(gals[p], E[q]))
                imp.append(max(top_k(gals[o], E[q]) for o in people if o != p))
        gen, imp = np.array(gen), np.array(imp)
        r = {"genuine_mean": float(gen.mean()), "genuine_min": float(gen.min()), "impostor_max": float(imp.max())}
        for lvl, t in TAU.items():
            r[f"{lvl}/tar"] = float(np.mean(gen >= t))
            r[f"{lvl}/far"] = float(np.mean(imp >= t))
            r[f"{lvl}/false_accepts"] = int(np.sum(imp >= t))
        out[name] = r
    # poisoning: every frame of every other participant, in time order, offered to each template
    poisoned, imp_gate_max, offered = 0, -1.0, 0
    for p in people:
        others = np.where(person != p)[0]
        others = others[np.argsort(day[others], kind="stable")]
        offered += len(others)
        for i in others:
            if ok[i]:
                imp_gate_max = max(imp_gate_max, top_k(static[p], E[i]))
        a, _ = adapt(static[p], [(E[i], ok[i], day[i]) for i in others], tau, rate_limit=False)
        poisoned += len(a)
    out["poisoning"] = {"impostor_frames_offered": int(offered), "impostor_samples_added": int(poisoned),
                        "impostor_top3_max": float(imp_gate_max), "update_gate": tau + MARGIN}
    return out


def main():
    items = exp_mpii.sample()
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init) as pool:
        res = list(pool.map(_embed, [str(f) for _, _, _, f in items], chunksize=16))
    print(f"embedded {len(items)} images in {time.time() - t0:.0f}s")
    keep = [i for i, r in enumerate(res) if r is not None]
    E = np.stack([res[i][0] for i in keep]).astype(np.float32)
    ok = np.array([res[i][1] for i in keep])
    person = np.array([items[i][0] for i in keep])
    day = np.array([items[i][1] for i in keep])
    ndays = np.array([items[i][2] for i in keep])
    out = {"seed": exp_mpii.SEED, "images": len(items), "detected": len(keep), "eligible_fraction": float(ok.mean()),
           "people": int(len(set(person))), "enroll": ENROLL, "margin": MARGIN, "novelty": NOVELTY}
    for lvl in ("balanced", "strict"):
        for mode, rl in (("one_per_day", True), ("no_rate_limit", False)):
            r = evaluate(E, ok, person, day, ndays, TAU[lvl], rl)
            out[f"{lvl}/{mode}"] = r
            s, a = r["static"], r["adapted"]
            print(f"{lvl:8s} {mode:14s} updates {r['updates_total']:4d} | genuine mean {s['genuine_mean']:.3f} -> "
                  f"{a['genuine_mean']:.3f}, min {s['genuine_min']:.3f} -> {a['genuine_min']:.3f} | strict TAR "
                  f"{100 * s['strict/tar']:.2f} -> {100 * a['strict/tar']:.2f}% | balanced FAR {100 * s['balanced/far']:.3f} -> "
                  f"{100 * a['balanced/far']:.3f}% | impostor max {s['impostor_max']:.3f} -> {a['impostor_max']:.3f} | "
                  f"poisoned {r['poisoning']['impostor_samples_added']} (impostor top-3 max {r['poisoning']['impostor_top3_max']:.3f})")
    save_json("adaptation", out)


if __name__ == "__main__":
    main()
