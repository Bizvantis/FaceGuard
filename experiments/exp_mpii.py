"""Recognition on real laptop-webcam images: MPIIFaceGaze (15 people, everyday laptop use over days to weeks).

Data: Zhang et al., MPIIFaceGaze (CC BY-NC-SA 4.0), data/mpiifacegaze/MPIIFaceGaze/pXX/dayYY/*.jpg,
1280x720 frames from each participant's own laptop webcam.

Sampling (seed 2026): up to 300 images per participant, spread evenly over the recording days.
Every image goes through the same pipeline as the app (YuNet, centre-most face, five-landmark alignment,
flip TTA); SFace and, as same-protocol baselines, the two ArcFace models of exp_baselines.py.

A  Pairwise verification: all same-person pairs taken on DIFFERENT days (cross-session) and all different-person
   pairs -> AUC, EER, TAR@FAR, FAR/FRR at the three FaceGuard thresholds (SFace).
B  FaceGuard enrollment protocol: each person enrolled with 22 images from their first recording days, probed
   with images from the later half of their days (top-3 open-set rule); impostors by leave-one-person-out
   (each person's probes against the gallery of the other 14).
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import exp_baselines as eb
from common import DATA, save_json

from faceguard.recognition.analysis import auc_score, equal_error_rate, roc_curve, tar_at_far
from faceguard.vision.camera import load_image
from faceguard.vision.embedder import normalize

ROOT = DATA / "mpiifacegaze" / "MPIIFaceGaze"
PER_PERSON, SEED, ENROLL, K = 300, 2026, 22, 3
THRESHOLDS = {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52}


def _embed(path):
    img = load_image(path)
    faces = eb._ENG.detector.detect(img)
    if not faces:
        return None
    h, w = img.shape[:2]
    face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
    _, sface, _ = eb._ENG.embed_face(img, face, tta=True)
    crop = eb.arc_align(img, face.landmarks)
    out = {k: eb.arc_embed(s, crop) for k, s in eb._SESS.items()}
    out["sface"] = normalize(np.asarray(sface, np.float32).reshape(-1))
    out["face_px"] = float(face.width)
    return out


def sample():
    rng = random.Random(SEED)
    items = []
    for pdir in sorted(ROOT.glob("p[0-9][0-9]")):
        days = sorted(d for d in pdir.glob("day*") if d.is_dir())
        files = [(di, f) for di, d in enumerate(days) for f in sorted(d.glob("*.jpg"))]
        pick = sorted(rng.sample(range(len(files)), min(PER_PERSON, len(files))))  # uniform over time
        items += [(pdir.name, files[i][0], len(days), files[i][1]) for i in pick]
    return items


def verification(E, person, day, same_mask_fn):
    sims = E @ E.T
    iu = np.triu_indices(len(E), 1)
    s = sims[iu]
    same = person[iu[0]] == person[iu[1]]
    cross_day = day[iu[0]] != day[iu[1]]
    gen, imp = s[same & cross_day], s[~same]
    far, tar, _ = roc_curve(gen, imp)
    eer, _ = equal_error_rate(gen, imp)
    out = {"genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)), "auc": auc_score(far, tar), "eer": eer,
           "tar_at_far_1e-2": tar_at_far(gen, imp, 1e-2)[0], "tar_at_far_1e-3": tar_at_far(gen, imp, 1e-3)[0],
           "tar_at_far_1e-4": tar_at_far(gen, imp, 1e-4)[0], "genuine_mean": float(gen.mean()), "impostor_mean": float(imp.mean()),
           "impostor_max": float(imp.max())}
    for name, t in THRESHOLDS.items():
        out[f"{name}/far"] = float(np.mean(imp >= t))
        out[f"{name}/frr"] = float(np.mean(gen < t))
        out[f"{name}/false_accepts"] = int(np.sum(imp >= t))
    return out


def enrollment_protocol(E, person, day, ndays):
    people = sorted(set(person))
    gal, probes = {}, {}
    for p in people:
        idx = np.where(person == p)[0]
        order = idx[np.argsort(day[idx], kind="stable")]
        gal[p] = order[:ENROLL]
        late = day[idx] >= ndays[idx] / 2.0  # second half of this person's recording days
        probes[p] = np.setdiff1d(idx[late], gal[p])

    def score(q, p):
        top = np.sort(E[gal[p]] @ E[q])[::-1][:K]
        return float(top.mean())
    out = {"people": len(people), "enroll_images": ENROLL}
    gen_scores, imp_best = [], []
    for p in people:
        for q in probes[p]:
            gen_scores.append(score(q, p))
            imp_best.append(max(score(q, o) for o in people if o != p))  # leave-one-person-out impostor
    gen, imp = np.array(gen_scores), np.array(imp_best)
    out["probes"] = int(len(gen))
    for name, t in THRESHOLDS.items():
        out[f"{name}/tar"] = float(np.mean(gen >= t))
        out[f"{name}/far"] = float(np.mean(imp >= t))
        out[f"{name}/false_accepts"] = int(np.sum(imp >= t))
    out["genuine_median"], out["impostor_best_max"] = float(np.median(gen)), float(imp.max())
    return out


def main():
    items = sample()
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=eb._init) as pool:
        embs = list(pool.map(_embed, [str(f) for _, _, _, f in items], chunksize=16))
    print(f"embedded {len(items)} images in {time.time() - t0:.0f}s")
    keep = [i for i, e in enumerate(embs) if e is not None]
    person = np.array([items[i][0] for i in keep])
    day = np.array([items[i][1] for i in keep])
    ndays = np.array([items[i][2] for i in keep])
    out = {"seed": SEED, "per_person": PER_PERSON, "images": len(items), "detected": len(keep),
           "people": int(len(set(person))), "face_px_median": float(np.median([embs[i]["face_px"] for i in keep])),
           "days_per_person": {p: int(n) for p, n in sorted({items[i][0]: items[i][2] for i in keep}.items())}}
    for model in ("sface", "arcface_mbf", "arcface_r50"):
        E = np.stack([embs[i][model] for i in keep]).astype(np.float32)
        out[model] = {"verification": verification(E, person, day, None), "enrollment": enrollment_protocol(E, person, day, ndays)}
        v, e = out[model]["verification"], out[model]["enrollment"]
        print(f"{model:12s} AUC {v['auc']:.4f} EER {100 * v['eer']:.2f}% TAR@1e-3 {100 * v['tar_at_far_1e-3']:.2f}% | "
              f"balanced FAR {100 * v['balanced/far']:.4f}% FRR {100 * v['balanced/frr']:.2f}% | enroll TAR {100 * e['balanced/tar']:.2f}% "
              f"FAR {100 * e['balanced/far']:.3f}%")
    save_json("mpii", out)


if __name__ == "__main__":
    main()
