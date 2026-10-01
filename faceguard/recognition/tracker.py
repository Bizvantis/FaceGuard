"""Lightweight multi-face tracker.

Associates detections across frames (greedy IoU, with a centre-distance
fallback for fast motion or low frame rates) so that identity evidence can be
accumulated per person instead of flickering frame to frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from faceguard.recognition.gallery import PENDING, MatchResult
from faceguard.vision.detector import Face
from faceguard.vision.geometry import iou
from faceguard.vision.mesh import MeshFace
from faceguard.vision.quality import FaceQuality


@dataclass
class Track:
    id: int
    face: Face
    first_seen: float
    last_seen: float
    hits: int = 1
    missed: int = 0
    embedding: np.ndarray | None = None
    aligned: np.ndarray | None = None
    quality: FaceQuality | None = None
    embedded_at: float = -1e9
    embed_count: int = 0
    score_ema: np.ndarray | None = None
    ema_version: int = -1
    match: MatchResult = PENDING
    mesh: MeshFace | None = field(default=None, repr=False)

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 5, max_age: float = 1.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_age = max_age
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []

    def _affinity(self, track: Track, face: Face) -> float:
        overlap = iou(track.face.bbox, face.bbox)
        if overlap >= self.iou_threshold:
            return overlap
        (tx, ty), (fx, fy) = track.face.center, face.center
        size = max(track.face.width, face.width, 1e-6)
        ratio = face.width / max(track.face.width, 1e-6)
        dist = math.hypot(tx - fx, ty - fy)
        if dist < 0.6 * size and 0.66 < ratio < 1.5:
            return 0.2 * (1.0 - dist / (0.6 * size))  # always ranks below a real IoU match
        return 0.0

    def update(self, faces: list[Face], t: float) -> list[Track]:
        pairs = []
        for ti, track in enumerate(self._tracks):
            for fi, face in enumerate(faces):
                a = self._affinity(track, face)
                if a > 0:
                    pairs.append((a, ti, fi))
        pairs.sort(key=lambda p: -p[0])
        used_t: set[int] = set()
        used_f: set[int] = set()
        for _, ti, fi in pairs:
            if ti in used_t or fi in used_f:
                continue
            track = self._tracks[ti]
            track.face = faces[fi]
            track.last_seen = t
            track.hits += 1
            track.missed = 0
            used_t.add(ti)
            used_f.add(fi)
        for ti, track in enumerate(self._tracks):
            if ti not in used_t:
                track.missed += 1
                track.mesh = None
        self._tracks = [
            tr for tr in self._tracks if tr.missed <= self.max_missed and t - tr.last_seen <= self.max_age
        ]
        for fi, face in enumerate(faces):
            if fi not in used_f:
                self._tracks.append(Track(self._next_id, face, t, t))
                self._next_id += 1
        active = [tr for tr in self._tracks if tr.missed == 0]
        active.sort(key=lambda tr: -tr.face.area)
        return active
