"""Machine-learning analysis of the enrolled gallery ("Model Lab").

* Score distributions: cosine similarity of every same-person pair (genuine)
  and every different-person pair (impostor), ROC AUC and equal error rate.
* Leave-one-out identification accuracy for several classifiers on top of the
  deep embeddings: FaceGuard's top-k cosine matcher, 1-NN, nearest centroid,
  and (with scikit-learn) a linear SVM, k-NN and logistic regression.
* A 2-D map of the embedding space (PCA, or t-SNE with scikit-learn).
* A data-driven threshold recommendation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from faceguard.recognition.database import SOURCE_ADAPTIVE, Person
from faceguard.vision.embedder import normalize


@dataclass
class PersonStats:
    name: str
    samples: int
    adaptive: int
    consistency: float  # mean cosine similarity of samples to the person's centroid
    worst: float  # lowest sample-to-centroid similarity
    nearest_other: str | None
    nearest_other_sim: float


@dataclass
class AnalysisReport:
    persons: list[PersonStats]
    genuine: np.ndarray
    impostor: np.ndarray
    methods: dict[str, float]
    auc: float | None
    eer: float | None
    eer_threshold: float | None
    recommended_threshold: float
    current_threshold: float
    projection: np.ndarray
    labels: np.ndarray
    names: list[str]
    projection_method: str
    notes: list[str] = field(default_factory=list)
    elapsed: float = 0.0


# ---------------------------------------------------------------- metrics
def roc_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(false accept rate, true accept rate, thresholds) for 'accept if score >= threshold'."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    distinct = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tar = np.r_[0.0, tp[distinct] / max(1, len(genuine))]
    far = np.r_[0.0, fp[distinct] / max(1, len(impostor))]
    thr = np.r_[np.inf, scores[distinct]]
    return far, tar, thr


def auc_score(far: np.ndarray, tar: np.ndarray) -> float:
    return float(np.trapezoid(tar, far)) if hasattr(np, "trapezoid") else float(np.trapz(tar, far))


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) where false accepts == false rejects."""
    far, tar, thr = roc_curve(genuine, impostor)
    frr = 1.0 - tar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0), float(thr[i] if np.isfinite(thr[i]) else 1.0)


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> tuple[float, float]:
    """Best true-accept rate with FAR <= target, and the threshold that achieves it."""
    far, tar, thr = roc_curve(genuine, impostor)
    ok = np.where(far <= target_far)[0]
    i = int(ok[-1])
    return float(tar[i]), float(thr[i] if np.isfinite(thr[i]) else 1.0)


def pair_scores(embeddings: np.ndarray, labels: np.ndarray, max_pairs: int = 200_000,
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarities of all same-label pairs and different-label pairs (subsampled if huge)."""
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    values = sims[iu]
    genuine, impostor = values[same], values[~same]
    rng = rng or np.random.default_rng(0)
    if len(impostor) > max_pairs:
        impostor = rng.choice(impostor, max_pairs, replace=False)
    if len(genuine) > max_pairs:
        genuine = rng.choice(genuine, max_pairs, replace=False)
    return genuine.astype(np.float32), impostor.astype(np.float32)


def recommend_threshold(genuine: np.ndarray, impostor: np.ndarray, floor: float = 0.36) -> tuple[float, list[str]]:
    notes: list[str] = []
    if len(impostor) == 0:
        return 0.45, ["Enroll a second person to measure impostor scores; using the balanced default."]
    worst_impostor = float(np.quantile(impostor, 0.999))
    rec = round(max(floor, worst_impostor + 0.05), 2)
    if len(genuine):
        low_genuine = float(np.quantile(genuine, 0.10))
        if rec > low_genuine:
            notes.append(
                "Some enrolled people look alike or some samples are poor; re-enrolling in even light will help."
            )
    return rec, notes


# ---------------------------------------------------------------- classifiers
def _topk_scores(sims_row: np.ndarray, labels: np.ndarray, n_classes: int, k: int) -> np.ndarray:
    out = np.full(n_classes, -1.0)
    for c in range(n_classes):
        s = sims_row[labels == c]
        if s.size:
            kk = min(k, s.size)
            out[c] = np.sort(s)[-kk:].mean()
    return out


def loo_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 3) -> dict[str, float]:
    """Leave-one-out accuracy of the similarity-based matchers (no training needed)."""
    n_classes = int(labels.max()) + 1
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    topk_ok = nn_ok = cen_ok = 0
    sums = np.stack([embeddings[labels == c].sum(axis=0) for c in range(n_classes)])
    counts = np.bincount(labels, minlength=n_classes)
    for i in range(len(labels)):
        y = labels[i]
        row = sims[i]
        mask = np.isfinite(row)
        topk_ok += int(np.argmax(_topk_scores(row[mask], labels[mask], n_classes, k)) == y)
        nn_ok += int(labels[int(np.argmax(row))] == y)
        cen = sums.copy()
        cen[y] -= embeddings[i]
        valid = counts - (np.arange(n_classes) == y) > 0
        cen = normalize(cen)
        cs = cen @ embeddings[i]
        cs[~valid] = -np.inf
        cen_ok += int(np.argmax(cs) == y)
    n = len(labels)
    return {
        f"Top-{k} cosine (FaceGuard)": topk_ok / n,
        "1-NN cosine": nn_ok / n,
        "Nearest centroid": cen_ok / n,
    }


def sklearn_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Stratified k-fold accuracy of trained classifiers (needs scikit-learn)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
    except ImportError:
        return {}
    min_count = int(np.bincount(labels).min())
    if min_count < 2:
        return {}
    cv = StratifiedKFold(n_splits=min(5, min_count), shuffle=True, random_state=0)
    models = {
        "Linear SVM": LinearSVC(C=1.0),
        "k-NN (k=5, cosine)": KNeighborsClassifier(n_neighbors=min(5, min_count), metric="cosine"),
        "Logistic regression": LogisticRegression(C=10.0, max_iter=2000),
    }
    out = {}
    for name, model in models.items():
        try:
            out[name] = float(cross_val_score(model, embeddings, labels, cv=cv).mean())
        except ValueError:
            continue
    return out


def project_2d(embeddings: np.ndarray, method: str = "pca") -> tuple[np.ndarray, str]:
    if len(embeddings) < 3:
        return np.zeros((len(embeddings), 2), dtype=np.float32), "pca"
    if method == "tsne" and len(embeddings) >= 10:
        try:
            from sklearn.manifold import TSNE

            perplexity = float(max(2.0, min(30.0, (len(embeddings) - 1) / 3.0)))
            emb2 = TSNE(n_components=2, perplexity=perplexity, init="pca", metric="cosine",
                        random_state=0).fit_transform(embeddings)
            return emb2.astype(np.float32), "t-SNE"
        except ImportError:
            pass
    centered = embeddings - embeddings.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(np.float32), "PCA"


# ---------------------------------------------------------------- main entry
def analyze(persons: list[Person], current_threshold: float, k: int = 3, projection: str = "pca") -> AnalysisReport:
    start = time.perf_counter()
    persons = [p for p in persons if p.samples]
    names = [p.name for p in persons]
    if not persons:
        return AnalysisReport([], np.zeros(0), np.zeros(0), {}, None, None, None, 0.45, current_threshold,
                              np.zeros((0, 2)), np.zeros(0, dtype=int), [], "PCA", ["Nobody is enrolled yet."])
    embs = normalize(np.vstack([p.embeddings() for p in persons]))
    labels = np.concatenate([np.full(p.count(), i) for i, p in enumerate(persons)]).astype(np.int32)
    centroids = normalize(np.stack([embs[labels == i].mean(axis=0) for i in range(len(persons))]))

    stats = []
    for i, p in enumerate(persons):
        sims = embs[labels == i] @ centroids[i]
        other, other_sim = None, 0.0
        if len(persons) > 1:
            cs = centroids @ centroids[i]
            cs[i] = -np.inf
            j = int(np.argmax(cs))
            other, other_sim = persons[j].name, float(cs[j])
        stats.append(PersonStats(p.name, p.count(), p.count(SOURCE_ADAPTIVE), float(sims.mean()),
                                 float(sims.min()), other, other_sim))

    genuine, impostor = pair_scores(embs, labels)
    methods: dict[str, float] = {}
    auc = eer = eer_thr = None
    notes: list[str] = []
    if len(persons) >= 2:
        methods.update(loo_accuracy(embs, labels, k))
        methods.update(sklearn_accuracy(embs, labels))
        if len(genuine) and len(impostor):
            far, tar, _ = roc_curve(genuine, impostor)
            auc = auc_score(far, tar)
            eer, eer_thr = equal_error_rate(genuine, impostor)
    else:
        notes.append("Enroll at least two people to measure identification accuracy and impostor scores.")
    rec, rec_notes = recommend_threshold(genuine, impostor)
    if len(persons) >= 2:
        notes.extend(rec_notes)
    for s in stats:
        if s.consistency < 0.6:
            notes.append(f"{s.name}'s samples are inconsistent (mean {s.consistency:.2f}); consider re-enrolling.")
    proj, method_name = project_2d(embs, projection)
    return AnalysisReport(stats, genuine, impostor, methods, auc, eer, eer_thr, rec, current_threshold,
                          proj, labels, names, method_name, notes, time.perf_counter() - start)
