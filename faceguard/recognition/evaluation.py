"""Benchmarking the recognition pipeline on labelled datasets.

Supported datasets
------------------
* any folder laid out as ``root/<person name>/<images>``
* ``olivetti`` - AT&T/Olivetti faces (40 people x 10 images, fetched by scikit-learn, ~1.4 MB)
* ``lfw``      - Labeled Faces in the Wild, standard 10-fold pair-verification protocol
                 (6000 pairs; scikit-learn downloads ~230 MB on first use)

Metrics: closed-set rank-1 identification (k-fold), open-set detection &
identification rate vs false-alarm rate, verification ROC AUC / EER /
TAR@FAR, accuracy at FaceGuard's configured threshold, and speed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from faceguard.recognition.analysis import auc_score, equal_error_rate, pair_scores, roc_curve, tar_at_far
from faceguard.recognition.engine import FaceEngine
from faceguard.recognition.gallery import Gallery, PersonInfo
from faceguard.vision.camera import list_images, load_image
from faceguard.vision.embedder import normalize

log = logging.getLogger(__name__)
Progress = Callable[[int, int], None]


@dataclass
class EvalReport:
    dataset: str
    identities: int
    images: int
    failures: int
    threshold: float
    closed_set_accuracy: float | None = None
    open_set_dir: float | None = None  # enrolled probes correctly identified above threshold
    open_set_far: float | None = None  # non-enrolled probes wrongly accepted
    auc: float | None = None
    eer: float | None = None
    eer_threshold: float | None = None
    tar_at_far: dict[str, float] = field(default_factory=dict)
    tar_at_threshold: float | None = None
    far_at_threshold: float | None = None
    lfw_accuracy: float | None = None
    lfw_std: float | None = None
    ms_per_image: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"  # noqa: E731
        lines = [
            f"Dataset            : {self.dataset}",
            f"Identities / images: {self.identities} / {self.images} (no face found in {self.failures})",
        ]
        if self.lfw_accuracy is not None:
            lines.append(f"LFW 10-fold acc.   : {pct(self.lfw_accuracy)} +/- {100 * (self.lfw_std or 0):.2f}")
        if self.closed_set_accuracy is not None:
            lines.append(f"Rank-1 accuracy    : {pct(self.closed_set_accuracy)}  (closed set, k-fold)")
        if self.open_set_dir is not None:
            lines.append(f"Open-set @ {self.threshold:.2f}  : identified {pct(self.open_set_dir)}, false alarms {pct(self.open_set_far)}")
        if self.auc is not None:
            lines.append(f"Verification AUC   : {self.auc:.5f}   EER {pct(self.eer)} at {self.eer_threshold:.3f}")
        for k, v in self.tar_at_far.items():
            lines.append(f"TAR @ FAR={k:<7}: {pct(v)}")
        if self.tar_at_threshold is not None:
            lines.append(f"Pairs @ {self.threshold:.2f}      : TAR {pct(self.tar_at_threshold)}, FAR {pct(self.far_at_threshold)}")
        lines.append(f"Speed              : {self.ms_per_image:.1f} ms per image (detect + align + embed)")
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------- embedding
def embed_image(engine: FaceEngine, image: np.ndarray, prefer_center: bool = False) -> np.ndarray | None:
    faces = engine.detector.detect(image)
    if not faces:
        return None
    if prefer_center:
        h, w = image.shape[:2]
        face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
    else:
        face = faces[0]
    _, emb, _ = engine.embed_face(image, face, tta=True)
    return emb


def embed_items(engine: FaceEngine, items: Iterable[tuple[np.ndarray | str | Path, int]], total: int,
                progress: Progress | None = None, prefer_center: bool = False) -> tuple[np.ndarray, np.ndarray, int, float]:
    embs, labels, failures = [], [], 0
    start = time.perf_counter()
    for i, (item, label) in enumerate(items):
        img = item if isinstance(item, np.ndarray) else load_image(item)
        emb = embed_image(engine, img, prefer_center) if img is not None else None
        if emb is None:
            failures += 1
        else:
            embs.append(emb)
            labels.append(label)
        if progress:
            progress(i + 1, total)
    elapsed = time.perf_counter() - start
    e = np.stack(embs) if embs else np.zeros((0, 128), np.float32)
    return e, np.asarray(labels, dtype=np.int32), failures, 1000.0 * elapsed / max(1, total)


# ---------------------------------------------------------------- protocols
def evaluate_embeddings(name: str, embs: np.ndarray, labels: np.ndarray, threshold: float, k: int = 3,
                        failures: int = 0, ms: float = 0.0, seed: int = 0) -> EvalReport:
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    rep = EvalReport(name, len(ids), len(labels) + failures, failures, threshold, ms_per_image=ms)
    if len(ids) < 2:
        rep.notes.append("Need at least two identities for a meaningful evaluation.")
        return rep

    # Closed-set identification: each image is a probe against a gallery of all *other* images.
    persons = [PersonInfo(str(i), str(i)) for i in ids]
    correct = total = 0
    folds = 5
    fold_of = np.empty(len(labels), dtype=np.int32)
    for c in ids:
        idx = np.where(labels == c)[0]
        fold_of[idx] = rng.permutation(len(idx)) % folds
    for f in range(folds):
        probe = fold_of == f
        gal = ~probe
        present = [c for c in ids if np.any(gal & (labels == c))]
        if len(present) < 2 or not probe.any():
            continue
        g = Gallery([persons[int(np.where(ids == c)[0][0])] for c in present],
                    [embs[gal & (labels == c)] for c in present])
        scores = g.score_matrix(embs[probe], k)
        pred = np.array(present)[np.argmax(scores, axis=1)]
        mask = np.isin(labels[probe], present)
        correct += int(np.sum(pred[mask] == labels[probe][mask]))
        total += int(mask.sum())
    rep.closed_set_accuracy = correct / total if total else None

    # Open-set: enroll half of the identities (half of their images), probe with the rest.
    perm = rng.permutation(ids)
    enrolled, others = perm[: len(ids) // 2], perm[len(ids) // 2:]
    gal_embs, gal_people, probes, probe_labels = [], [], [], []
    for c in enrolled:
        idx = rng.permutation(np.where(labels == c)[0])
        if len(idx) < 2:
            continue
        half = max(1, len(idx) // 2)
        gal_embs.append(embs[idx[:half]])
        gal_people.append(PersonInfo(str(c), str(c)))
        probes.extend(embs[idx[half:]])
        probe_labels.extend([c] * (len(idx) - half))
    impostor_probes = embs[np.isin(labels, others)]
    if gal_embs and probes and len(impostor_probes):
        g = Gallery(gal_people, gal_embs)
        s_gen = g.score_matrix(np.stack(probes), k)
        best = np.argmax(s_gen, axis=1)
        ok = (np.array([int(p.id) for p in gal_people])[best] == np.array(probe_labels)) & (s_gen.max(axis=1) >= threshold)
        rep.open_set_dir = float(ok.mean())
        rep.open_set_far = float((g.score_matrix(impostor_probes, k).max(axis=1) >= threshold).mean())

    # Verification over image pairs.
    genuine, impostor = pair_scores(embs, labels, rng=rng)
    if len(genuine) and len(impostor):
        far, tar, _ = roc_curve(genuine, impostor)
        rep.auc = auc_score(far, tar)
        rep.eer, rep.eer_threshold = equal_error_rate(genuine, impostor)
        for target in (1e-2, 1e-3):
            if len(impostor) * target >= 1:
                rep.tar_at_far[f"{target:g}"] = tar_at_far(genuine, impostor, target)[0]
        rep.tar_at_threshold = float(np.mean(genuine >= threshold))
        rep.far_at_threshold = float(np.mean(impostor >= threshold))
    return rep


def evaluate_folder(engine: FaceEngine, root: str | Path, threshold: float, progress: Progress | None = None) -> EvalReport:
    root = Path(root)
    people = sorted(p for p in root.iterdir() if p.is_dir())
    items = [(f, i) for i, p in enumerate(people) for f in list_images(p)]
    embs, labels, failures, ms = embed_items(engine, items, len(items), progress)
    return evaluate_embeddings(root.name, normalize(embs), labels, threshold, failures=failures, ms=ms)


def olivetti_items(data_home: str | None = None) -> list[tuple[np.ndarray, int]]:
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces(data_home=data_home, shuffle=False)
    items = []
    for img, label in zip(data.images, data.target):
        g = (img * 255).astype(np.uint8)
        # The 64x64 crops are too tight for a detector: pad and upscale them first.
        g = cv2.resize(cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_REPLICATE), None, fx=3, fy=3,
                       interpolation=cv2.INTER_CUBIC)
        items.append((cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), int(label)))
    return items


def evaluate_olivetti(engine: FaceEngine, threshold: float, data_home: str | None = None,
                      progress: Progress | None = None) -> EvalReport:
    items = olivetti_items(data_home)
    embs, labels, failures, ms = embed_items(engine, items, len(items), progress)
    rep = evaluate_embeddings("AT&T / Olivetti faces", normalize(embs), labels, threshold, failures=failures, ms=ms)
    rep.notes.append("Olivetti images are 64x64 grayscale; impostor scores run higher than on webcam colour images.")
    return rep


def evaluate_lfw(engine: FaceEngine, data_home: str | None = None, progress: Progress | None = None) -> EvalReport:
    """Standard LFW protocol: 10 folds x 600 pairs; threshold tuned on 9 folds, tested on the 10th."""
    from sklearn.datasets._lfw import _check_fetch_lfw

    lfw_home, folder = _check_fetch_lfw(data_home=data_home, funneled=True, download_if_missing=True)
    lines = Path(lfw_home, "pairs.txt").read_text().split("\n")
    n_folds, n_per = (int(v) for v in lines[0].split())
    pairs = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            pairs.append(((parts[0], int(parts[1])), (parts[0], int(parts[2])), 1))
        elif len(parts) == 4:
            pairs.append(((parts[0], int(parts[1])), (parts[2], int(parts[3])), 0))
    cache: dict[tuple[str, int], np.ndarray | None] = {}
    needed = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    start = time.perf_counter()
    for i, (name, idx) in enumerate(needed):
        img = load_image(Path(folder, name, f"{name}_{idx:04d}.jpg"))
        cache[(name, idx)] = embed_image(engine, img, prefer_center=True) if img is not None else None
        if progress:
            progress(i + 1, len(needed))
    ms = 1000.0 * (time.perf_counter() - start) / max(1, len(needed))
    failures = sum(v is None for v in cache.values())
    sims = np.array([
        float(cache[a] @ cache[b]) if cache[a] is not None and cache[b] is not None else -1.0 for a, b, _ in pairs
    ])
    same = np.array([s for _, _, s in pairs], dtype=bool)
    fold = np.repeat(np.arange(n_folds), 2 * n_per)[: len(pairs)]
    candidates = np.linspace(-0.2, 1.0, 1201)
    accs = []
    for f in range(n_folds):
        train, test = fold != f, fold == f
        train_acc = [np.mean((sims[train] >= c) == same[train]) for c in candidates]
        thr = candidates[int(np.argmax(train_acc))]
        accs.append(np.mean((sims[test] >= thr) == same[test]))
    rep = EvalReport("LFW (10-fold pairs)", len({n for n, _ in needed}), len(needed), failures, 0.0, ms_per_image=ms)
    rep.lfw_accuracy, rep.lfw_std = float(np.mean(accs)), float(np.std(accs))
    far, tar, _ = roc_curve(sims[same], sims[~same])
    rep.auc = auc_score(far, tar)
    rep.eer, rep.eer_threshold = equal_error_rate(sims[same], sims[~same])
    return rep


def save_report(report: EvalReport, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    (path.with_suffix(".txt")).write_text(report.summary(), encoding="utf-8")
    return path
