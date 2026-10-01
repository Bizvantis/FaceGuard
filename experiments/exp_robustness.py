"""Robustness of verification to image degradations and occlusions (LFW View-2 folds 1-2, 1,200 pairs).

The enrolled (first) image of every pair stays clean; the probe (second) image is degraded before
detection, as happens when enrollment was done in good conditions and the live camera is not.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import key_index, lfw_embeddings, lfw_folder, lfw_pairs, make_engine, save_json

from faceguard.recognition.analysis import auc_score, roc_curve
from faceguard.vision.camera import load_image
from faceguard.vision.detector import Face
from faceguard.vision.embedder import normalize

CONDITIONS = (
    [("clean", 0)]
    + [("blur", s) for s in (1, 2, 3, 4, 6)]
    + [("gain", g) for g in (0.15, 0.3, 0.5, 1.5, 2.0, 3.0)]
    + [("scale", s) for s in (0.5, 0.35, 0.25, 0.18, 0.12)]
    + [("jpeg", q) for q in (30, 15, 8, 4)]
    + [("noise", s) for s in (8, 16, 24, 32)]
    + [("mask", 1), ("sunglasses", 1)]
)
THRESHOLD = 0.45
IMPOSTOR_GALLERY = 1000
IMPOSTOR_SEED = 17
_E = None
_MEAN = None


def _init(mean_row):
    global _E, _MEAN
    cv2.setNumThreads(2)
    _E, _MEAN = make_engine(), mean_row


def degrade(img, kind, level, face):
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), level)
    if kind == "gain":
        return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "scale":
        return cv2.resize(img, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
    if kind == "jpeg":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])[1], cv2.IMREAD_COLOR)
    if kind == "noise":
        rng = np.random.default_rng(int(img.sum()) % 2**32)
        return np.clip(img + rng.normal(0, level, img.shape), 0, 255).astype(np.uint8)
    out = img.copy()
    lm = face.landmarks
    iod = float(np.linalg.norm(lm[1] - lm[0]))
    if kind == "mask":  # a surgical-mask-like occluder from just below the nose to the chin
        x, y, w, h = face.bbox
        top = int(lm[2][1] + 0.15 * iod)
        cv2.rectangle(out, (int(x - 0.05 * w), top), (int(x + 1.05 * w), int(y + h + 0.1 * h)), (200, 180, 150), -1)
    elif kind == "sunglasses":
        for p in lm[:2]:
            cv2.ellipse(out, (int(p[0]), int(p[1])), (int(0.42 * iod), int(0.28 * iod)), 0, 0, 360, (20, 20, 20), -1)
        cv2.line(out, tuple(lm[0].astype(int)), tuple(lm[1].astype(int)), (20, 20, 20), max(2, int(0.08 * iod)))
    return out


def _probe(args):
    path, kind, level = args
    img = load_image(path)
    h, w = img.shape[:2]
    clean_faces = _E.detector.detect(img)
    center = lambda fs, W, H: min(fs, key=lambda f: (f.center[0] - W / 2) ** 2 + (f.center[1] - H / 2) ** 2)  # noqa: E731
    ref = center(clean_faces, w, h) if clean_faces else Face.from_row(_MEAN)
    x = degrade(img, kind, level, ref)
    H, W = x.shape[:2]
    faces = _E.detector.detect(x)
    detected = bool(faces)
    if faces:
        face = center(faces, W, H)
    else:
        row = _MEAN.copy()
        row[:14] *= W / w  # mean landmarks rescaled to the degraded image
        face = Face.from_row(row)
    aligned = _E.embedder.align(x, face)
    return _E.embedder.embed_aligned(aligned, tta=True), detected, float(face.width)


def main():
    data = lfw_embeddings()
    key = key_index(data)
    _, folder = lfw_folder()
    pairs, n_folds, n_per = lfw_pairs()
    pairs = pairs[: 2 * 2 * n_per]  # folds 1-2
    same = np.array([s for _, _, s in pairs], dtype=bool)
    gal = np.stack([data["e_tta"][key[a]] for a, _, _ in pairs])
    probe_paths = [folder / b[0] / f"{b[0]}_{b[1]:04d}.jpg" for _, b, _ in pairs]
    tasks = [(p, k, lv) for k, lv in CONDITIONS for p in probe_paths]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(data["mean_row"],)) as pool:
        res = list(pool.map(_probe, tasks, chunksize=40))
    print(f"embedded {len(tasks)} degraded probes in {time.time() - t0:.0f}s")
    # Large impostor set: every degraded probe against 1,000 clean images of other people (fixed seed),
    # so a "0% FAR" rests on ~1.2 million impostor comparisons per condition, not on 600 pairs.
    names = data["names"]
    probe_names = np.array([b[0] for _, b, _ in pairs])
    G_idx = np.random.default_rng(IMPOSTOR_SEED).choice(len(names), IMPOSTOR_GALLERY, replace=False)
    G = data["e_tta"][G_idx]
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    other = probe_names[:, None] != names[G_idx][None, :]
    out = {"pairs": len(pairs), "threshold": THRESHOLD, "impostor_gallery": IMPOSTOR_GALLERY,
           "impostor_seed": IMPOSTOR_SEED, "impostor_comparisons": int(other.sum()), "conditions": []}
    n = len(pairs)
    for c, (kind, level) in enumerate(CONDITIONS):
        chunk = res[c * n:(c + 1) * n]
        P = np.stack([normalize(r[0]) for r in chunk])
        sims = np.einsum("ij,ij->i", gal, P)
        imp = (P @ G.T)[other]
        far, tar, _ = roc_curve(sims[same], sims[~same])
        row = {"kind": kind, "level": level, "auc": auc_score(far, tar),
               "tar_at_thr": float(np.mean(sims[same] >= THRESHOLD)), "far_at_thr": float(np.mean(sims[~same] >= THRESHOLD)),
               "far_large": float(np.mean(imp >= THRESHOLD)), "false_accepts_large": int(np.sum(imp >= THRESHOLD)),
               "impostor_max_large": float(imp.max()),
               "acc_at_thr": float(np.mean((sims >= THRESHOLD) == same)),
               "genuine_mean": float(sims[same].mean()), "detected": float(np.mean([r[1] for r in chunk])),
               "face_px": float(np.median([r[2] for r in chunk]))}
        out["conditions"].append(row)
        print(f"{kind:10s} {level:>5}  AUC {row['auc']:.4f}  TAR@0.45 {100 * row['tar_at_thr']:6.2f}%  FAR {100 * row['far_at_thr']:.2f}%  "
              f"FAR(1.2M) {100 * row['far_large']:.5f}% ({row['false_accepts_large']})  max imp {row['impostor_max_large']:.3f}  "
              f"genuine {row['genuine_mean']:.3f}  detected {100 * row['detected']:.1f}%  face {row['face_px']:.0f}px")
    save_json("robustness", out)


if __name__ == "__main__":
    main()
