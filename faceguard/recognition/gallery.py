"""Open-set face identification against an enrolled gallery.

Each enrolled person has many embeddings (different poses / lighting).  A query
embedding is scored against every person as the mean of its ``k`` highest
cosine similarities to that person's samples ("top-k mean").  This is more
robust than a single centroid (it uses the samples whose pose matches the
query) and less noisy than a single nearest neighbour.

Open-set decision:

* ``match``     best score >= threshold
* ``unknown``   best score <  unknown_threshold  (confidently not enrolled)
* ``uncertain`` anything in between (never used to unlock or to raise alarms)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from faceguard.vision.embedder import EMBEDDING_DIM, normalize


@dataclass(frozen=True)
class PersonInfo:
    id: str
    name: str
    is_owner: bool = False
    trusted: bool = False

    @property
    def authorized(self) -> bool:
        """Owner and trusted people may use the PC without triggering the guard."""
        return self.is_owner or self.trusted


class Gallery:
    """Immutable snapshot of the enrolled embeddings, safe to share between threads."""

    def __init__(self, persons: Sequence[PersonInfo], embeddings: Sequence[np.ndarray], version: int = 0) -> None:
        mats = [normalize(np.asarray(e, dtype=np.float32).reshape(-1, EMBEDDING_DIM)) for e in embeddings]
        keep = [i for i, m in enumerate(mats) if len(m)]
        self.version = version
        self.persons: list[PersonInfo] = [persons[i] for i in keep]
        if keep:
            self.matrix = np.ascontiguousarray(np.vstack([mats[i] for i in keep]))
            self.labels = np.concatenate([np.full(len(mats[i]), j, dtype=np.int32) for j, i in enumerate(keep)])
            self.centroids = normalize(np.stack([mats[i].mean(axis=0) for i in keep]))
        else:
            self.matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self.labels = np.zeros(0, dtype=np.int32)
            self.centroids = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self.slices: list[tuple[int, int]] = []
        start = 0
        for i in keep:
            self.slices.append((start, start + len(mats[i])))
            start += len(mats[i])

    def __len__(self) -> int:
        return len(self.persons)

    @property
    def empty(self) -> bool:
        return not self.persons

    def index_of(self, person_id: str) -> int | None:
        for j, p in enumerate(self.persons):
            if p.id == person_id:
                return j
        return None

    @property
    def owner(self) -> PersonInfo | None:
        return next((p for p in self.persons if p.is_owner), None)

    def scores(self, embedding: np.ndarray, k: int = 3) -> np.ndarray:
        """Top-k mean cosine similarity of one embedding to every person -> shape (P,)."""
        if self.empty:
            return np.zeros(0, dtype=np.float32)
        return self.score_matrix(np.asarray(embedding, dtype=np.float32).reshape(1, -1), k)[0]

    def score_matrix(self, embeddings: np.ndarray, k: int = 3) -> np.ndarray:
        """Top-k mean similarity for a batch of embeddings -> shape (N, P)."""
        e = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        sims = e @ self.matrix.T  # (N, M)
        out = np.empty((len(e), len(self.persons)), dtype=np.float32)
        for j, (s, t) in enumerate(self.slices):
            seg = sims[:, s:t]
            kk = min(k, t - s)
            if kk < seg.shape[1]:
                seg = np.partition(seg, seg.shape[1] - kk, axis=1)[:, -kk:]
            out[:, j] = seg.mean(axis=1)
        return out


@dataclass(frozen=True)
class MatchResult:
    status: str  # "match" | "uncertain" | "unknown" | "pending"
    person: PersonInfo | None = None  # best-scoring person (also set when not a match, for logs)
    similarity: float = 0.0
    runner_up: float = 0.0

    @property
    def is_match(self) -> bool:
        return self.status == "match"

    @property
    def is_unknown(self) -> bool:
        return self.status == "unknown"

    @property
    def label(self) -> str:
        if self.status == "match" and self.person:
            return self.person.name
        if self.status == "unknown":
            return "Unknown"
        if self.status == "uncertain":
            return "Not sure"
        return "..."


PENDING = MatchResult("pending")


class Matcher:
    def __init__(self, threshold: float = 0.45, unknown_threshold: float = 0.30, top_k: int = 3, margin: float = 0.0):
        self.threshold = float(threshold)
        self.unknown_threshold = float(min(unknown_threshold, threshold))
        self.top_k = int(top_k)
        self.margin = float(margin)

    def decide(self, gallery: Gallery, scores: np.ndarray) -> MatchResult:
        if gallery.empty or scores.size == 0:
            return MatchResult("unknown")
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        s = float(scores[best])
        r = float(scores[order[1]]) if len(order) > 1 else 0.0
        if s >= self.threshold and s - r >= self.margin:
            status = "match"
        elif s < self.unknown_threshold:
            status = "unknown"
        else:
            status = "uncertain"
        return MatchResult(status, gallery.persons[best], s, r)

    def identify(self, gallery: Gallery, embedding: np.ndarray) -> MatchResult:
        return self.decide(gallery, gallery.scores(embedding, self.top_k))
