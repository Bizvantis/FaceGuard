"""Shared helpers for the experiments behind the FaceGuard paper.

Every number reported in the paper is produced by a script in this folder and
stored in ``experiments/results/*.json`` (figures in ``paper/figures``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faceguard.config import Settings  # noqa: E402
from faceguard.recognition.engine import FaceEngine  # noqa: E402
from faceguard.vision import quality as fq  # noqa: E402
from faceguard.vision.camera import load_image  # noqa: E402
from faceguard.vision.detector import Face  # noqa: E402
from faceguard.vision.embedder import normalize  # noqa: E402

EXP = Path(__file__).resolve().parent
DATA = EXP / "data"
RESULTS = EXP / "results"
FIGURES = ROOT / "paper" / "figures"
for _d in (RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

DET_THRESHOLD = 0.5  # benchmark detection threshold (the app uses 0.8 for live video)


def save_json(name: str, obj) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print(f"[saved] {path.relative_to(ROOT)}")
    return path


def load_json(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def make_engine(det_threshold: float = DET_THRESHOLD) -> FaceEngine:
    return FaceEngine.create(Settings(detection_score=det_threshold, min_face_size=20))


# ---------------------------------------------------------------- LFW
def lfw_folder() -> tuple[Path, Path]:
    from sklearn.datasets._lfw import _check_fetch_lfw

    home, folder = _check_fetch_lfw(data_home=str(DATA), funneled=False, download_if_missing=True)
    return Path(home), Path(folder)


def lfw_index() -> list[tuple[str, int, Path]]:
    _, folder = lfw_folder()
    items = []
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(person.glob("*.jpg")):
            idx = int(f.stem.rsplit("_", 1)[1])
            items.append((person.name, idx, f))
    return items


def lfw_pairs() -> tuple[list[tuple[tuple[str, int], tuple[str, int], int]], int, int]:
    home, _ = lfw_folder()
    lines = (home / "pairs.txt").read_text().split("\n")
    n_folds, n_per = (int(v) for v in lines[0].split())
    pairs = []
    for line in lines[1:]:
        p = line.split()
        if len(p) == 3:
            pairs.append(((p[0], int(p[1])), (p[0], int(p[2])), 1))
        elif len(p) == 4:
            pairs.append(((p[0], int(p[1])), (p[2], int(p[3])), 0))
    return pairs, n_folds, n_per


# ---------------------------------------------------------------- embedding workers
_ENGINE: FaceEngine | None = None


def _init_worker() -> None:
    global _ENGINE
    cv2.setNumThreads(2)
    _ENGINE = make_engine()


def square_crop(img: np.ndarray, face: Face, size: int = 112) -> np.ndarray:
    """Alignment-free baseline: the detector box made square, cropped and resized."""
    x, y, w, h = (float(v) for v in face.bbox)
    s = max(w, h)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = int(s)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y0 + pad: y0 + pad + int(s), x0 + pad: x0 + pad + int(s)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def describe_face(img: np.ndarray, fallback_row: np.ndarray | None = None, prefer_center: bool = True,
                  engine: FaceEngine | None = None) -> dict | None:
    """Detect (centre-most face), align, embed three ways, and measure pose / quality."""
    eng = engine or _ENGINE
    faces = eng.detector.detect(img)
    fallback = False
    if faces:
        h, w = img.shape[:2]
        face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2) if prefer_center else faces[0]
    elif fallback_row is not None:
        face, fallback = Face.from_row(fallback_row), True
    else:
        return None
    net = eng.embedder
    aligned = net.align(img, face)
    f1 = normalize(net._net.feature(aligned).reshape(-1))
    f2 = normalize(net._net.feature(cv2.flip(aligned, 1)).reshape(-1))
    crop = square_crop(img, face)
    c1 = normalize(net._net.feature(crop).reshape(-1))
    c2 = normalize(net._net.feature(cv2.flip(crop, 1)).reshape(-1))
    sharp, bright, contrast, over = fq.image_stats(aligned)
    return {
        "row": face.row,
        "fallback": fallback,
        "n_faces": len(faces),
        "yaw": face.pose.yaw,
        "pitch": face.pose.pitch,
        "roll": face.pose.roll,
        "sharp": sharp,
        "bright": bright,
        "e_single": f1,
        "e_tta": normalize(f1 + f2),
        "e_crop": normalize(c1 + c2),
    }


def _embed_chunk(args):
    paths, fallback_row = args
    out = []
    for p in paths:
        img = load_image(p)
        out.append(describe_face(img, fallback_row) if img is not None else None)
    return out


def embed_images(paths: list[Path], fallback_row: np.ndarray | None = None, workers: int = 4, chunk: int = 64) -> list[dict | None]:
    chunks = [(paths[i:i + chunk], fallback_row) for i in range(0, len(paths), chunk)]
    results: list[dict | None] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for i, part in enumerate(pool.map(_embed_chunk, chunks)):
            results.extend(part)
            if i % 20 == 0:
                print(f"  embedded {len(results)}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)
    return results


def lfw_embeddings(force: bool = False) -> dict:
    """All 13,233 LFW images, cached in results/lfw_embeddings.npz."""
    cache = RESULTS / "lfw_embeddings.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    index = lfw_index()
    paths = [p for _, _, p in index]
    print(f"Embedding {len(paths)} LFW images (pass 1)...")
    first = embed_images(paths)
    rows = np.array([r["row"] for r in first if r is not None and not r["fallback"]])
    mean_row = rows.mean(axis=0).astype(np.float32)  # average face position/landmarks = fallback for misses
    missing = [i for i, r in enumerate(first) if r is None]
    print(f"Detections: {len(paths) - len(missing)}/{len(paths)}; re-embedding {len(missing)} misses with the mean landmarks")
    if missing:
        redo = embed_images([paths[i] for i in missing], fallback_row=mean_row, workers=2)
        for i, r in zip(missing, redo):
            first[i] = r
    out = {
        "names": np.array([n for n, _, _ in index]),
        "idx": np.array([i for _, i, _ in index], dtype=np.int32),
        "fallback": np.array([r["fallback"] for r in first]),
        "n_faces": np.array([r["n_faces"] for r in first], dtype=np.int32),
        "yaw": np.array([r["yaw"] for r in first], dtype=np.float32),
        "pitch": np.array([r["pitch"] for r in first], dtype=np.float32),
        "roll": np.array([r["roll"] for r in first], dtype=np.float32),
        "sharp": np.array([r["sharp"] for r in first], dtype=np.float32),
        "bright": np.array([r["bright"] for r in first], dtype=np.float32),
        "face_w": np.array([r["row"][2] for r in first], dtype=np.float32),
        "e_single": np.stack([r["e_single"] for r in first]).astype(np.float32),
        "e_tta": np.stack([r["e_tta"] for r in first]).astype(np.float32),
        "e_crop": np.stack([r["e_crop"] for r in first]).astype(np.float32),
        "mean_row": mean_row,
    }
    np.savez_compressed(cache, **out)
    return out


def key_index(data: dict) -> dict[tuple[str, int], int]:
    return {(str(n), int(i)): k for k, (n, i) in enumerate(zip(data["names"], data["idx"]))}


def ten_fold_accuracy(sims: np.ndarray, same: np.ndarray, n_folds: int = 10) -> tuple[float, float, float]:
    """LFW View-2 protocol: threshold tuned on 9 folds, accuracy on the 10th. Returns mean, std, mean threshold."""
    fold = np.repeat(np.arange(n_folds), len(sims) // n_folds)
    cands = np.linspace(-0.2, 1.0, 1201)
    accs, thrs = [], []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        best = cands[int(np.argmax([np.mean((sims[tr] >= c) == same[tr]) for c in cands]))]
        thrs.append(best)
        accs.append(np.mean((sims[te] >= best) == same[te]))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


if __name__ == "__main__":
    data = lfw_embeddings(force="--force" in sys.argv)
    print({k: v.shape for k, v in data.items()})
