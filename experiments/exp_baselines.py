"""Same-protocol recognition baselines: ArcFace (InsightFace WebFace600K models) vs SFace on LFW View 2.

Both networks see the same YuNet detections (centre-most face, mean-landmark fallback), are aligned to
their own canonical 112x112 template from the same five landmarks, and use flip test-time augmentation.
Metrics follow exp_lfw.py (10-fold accuracy, AUC, EER, TAR@FAR) plus per-fold accuracies for paired tests.

Models (downloaded from github.com/deepinsight/insightface releases v0.7, non-commercial research licence):
  data/insightface/buffalo_sc/w600k_mbf.onnx  ArcFace loss, MobileFaceNet, WebFace600K   (13.6 MB)
  data/insightface/buffalo_l/w600k_r50.onnx   ArcFace loss, ResNet-50,    WebFace600K   (174.4 MB)

    python exp_baselines.py            # embeddings + verification metrics
    python exp_baselines.py --latency  # per-crop latency (run on an otherwise idle machine)
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import DATA, key_index, lfw_embeddings, lfw_folder, lfw_pairs, load_json, make_engine, save_json

from faceguard.recognition.analysis import auc_score, equal_error_rate, roc_curve, tar_at_far
from faceguard.vision.camera import load_image
from faceguard.vision.detector import Face

MODELS = {"arcface_mbf": DATA / "insightface" / "buffalo_sc" / "w600k_mbf.onnx",
          "arcface_r50": DATA / "insightface" / "buffalo_l" / "w600k_r50.onnx"}
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)  # InsightFace 112x112 template
_ENG = _SESS = None


def _init(threads=2):
    global _ENG, _SESS
    import onnxruntime as ort
    cv2.setNumThreads(1)
    _ENG = make_engine()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    _SESS = {k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in MODELS.items()}


def arc_align(img, landmarks):
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARC_DST, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


def arc_embed(sess, crop):
    blob = cv2.dnn.blobFromImages([crop, cv2.flip(crop, 1)], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    e = out[0] + out[1]  # flip TTA, as for SFace
    return e / np.linalg.norm(e)


def _embed(args):
    path, mean_row = args
    img = load_image(path)
    faces = _ENG.detector.detect(img)
    if faces:
        h, w = img.shape[:2]
        face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
    else:
        face = Face.from_row(mean_row)
    crop = arc_align(img, face.landmarks)
    return {k: arc_embed(s, crop) for k, s in _SESS.items()}


def per_fold_accuracy(sims, same, n_folds=10):
    """Same protocol as common.ten_fold_accuracy, returning the ten fold accuracies."""
    fold = np.repeat(np.arange(n_folds), len(sims) // n_folds)
    cands = np.linspace(-0.2, 1.0, 1201)
    accs = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        best = cands[int(np.argmax([np.mean((sims[tr] >= c) == same[tr]) for c in cands]))]
        accs.append(float(np.mean((sims[te] >= best) == same[te])))
    return accs


def metrics(sims, same):
    accs = per_fold_accuracy(sims, same)
    far, tar, _ = roc_curve(sims[same], sims[~same])
    eer, _ = equal_error_rate(sims[same], sims[~same])
    return {"accuracy": float(np.mean(accs)), "std": float(np.std(accs)), "fold_acc": accs, "auc": auc_score(far, tar),
            "eer": eer, "tar_at_far_1e-2": tar_at_far(sims[same], sims[~same], 1e-2)[0],
            "tar_at_far_1e-3": tar_at_far(sims[same], sims[~same], 1e-3)[0]}


def latency():
    """Per-crop embedding latency incl. flip TTA (2 crops), 200 runs after 20 warm-up runs."""
    _init(threads=0)  # onnxruntime default thread pool
    cv2.setNumThreads(-1)  # OpenCV default threading, as in exp_runtime.py
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    out = {}
    fns = {"sface": lambda: _ENG.embedder.embed_aligned(crop, tta=True)}
    fns.update({k: (lambda s=s: arc_embed(s, crop)) for k, s in _SESS.items()})
    for name, fn in fns.items():
        for _ in range(20):
            fn()
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            fn()
            ts.append(1000 * (time.perf_counter() - t0))
        out[name] = {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts))}
        print(f"{name:12s} {out[name]['mean_ms']:.1f} +/- {out[name]['std_ms']:.1f} ms (with flip TTA)")
    res = load_json("baselines")
    res["latency_tta_ms"] = out
    save_json("baselines", res)


def main():
    if "--latency" in sys.argv:
        return latency()
    data = lfw_embeddings()
    key = key_index(data)
    _, folder = lfw_folder()
    pairs, _, _ = lfw_pairs()
    same = np.array([s for _, _, s in pairs], dtype=bool)
    needed = sorted({key[a] for a, _, _ in pairs} | {key[b] for _, b, _ in pairs})
    paths = [folder / data["names"][i] / f"{data['names'][i]}_{int(data['idx'][i]):04d}.jpg" for i in needed]
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init) as pool:
        embs = list(pool.map(_embed, [(p, data["mean_row"]) for p in paths], chunksize=32))
    print(f"embedded {len(paths)} images with {len(MODELS)} models in {time.time() - t0:.0f}s")
    row = {i: k for k, i in enumerate(needed)}
    out = {"pairs": len(pairs), "images": len(needed), "models": {k: str(p.relative_to(DATA)) for k, p in MODELS.items()},
           "model_mb": {k: round(p.stat().st_size / 2**20, 1) for k, p in MODELS.items()}, "results": {}}
    E_sf = data["e_tta"]
    sims = {"sface": np.array([float(E_sf[key[a]] @ E_sf[key[b]]) for a, b, _ in pairs]),
            "sface_no_tta": np.array([float(data["e_single"][key[a]] @ data["e_single"][key[b]]) for a, b, _ in pairs])}
    for m in MODELS:
        sims[m] = np.array([float(embs[row[key[a]]][m] @ embs[row[key[b]]][m]) for a, b, _ in pairs])
    for name, s in sims.items():
        out["results"][name] = metrics(s, same)
        r = out["results"][name]
        print(f"{name:13s} acc {100 * r['accuracy']:.2f} +/- {100 * r['std']:.2f}  AUC {r['auc']:.4f}  EER {100 * r['eer']:.2f}%"
              f"  TAR@1e-3 {100 * r['tar_at_far_1e-3']:.2f}%")
    save_json("baselines", out)


if __name__ == "__main__":
    main()
