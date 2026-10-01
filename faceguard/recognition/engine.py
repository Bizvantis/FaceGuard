"""The per-frame face-recognition engine.

``FaceEngine.process(frame)`` runs::

    detect (YuNet) -> track (IoU) -> align + embed (SFace, throttled per track)
      -> score vs gallery (top-k cosine, temporally smoothed) -> open-set decision
      -> optional dense mesh (MediaPipe) for liveness / visualisation

It is not thread-safe: create one engine per worker thread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from faceguard import models
from faceguard.config import Settings
from faceguard.recognition.gallery import Gallery, Matcher, MatchResult
from faceguard.recognition.tracker import FaceTracker, Track
from faceguard.vision import quality as fq
from faceguard.vision.detector import Face, FaceDetector
from faceguard.vision.embedder import FaceEmbedder
from faceguard.vision.geometry import HeadPose, iou
from faceguard.vision.mesh import MeshAnalyzer, MeshFace, mediapipe_available

log = logging.getLogger(__name__)


@dataclass
class EngineOptions:
    recognize: bool = True
    mesh: bool = False
    embed_interval: float = 0.35  # seconds between re-embeddings of the same track
    max_embeds_per_frame: int = 4
    quality: fq.QualityThresholds = fq.PASSIVE_THRESHOLDS


@dataclass
class FaceResult:
    """Per-face output for one frame (a detached copy; safe to hand to the GUI thread)."""

    track_id: int
    bbox: np.ndarray
    landmarks: np.ndarray
    det_score: float
    pose: HeadPose
    match: MatchResult
    quality: fq.FaceQuality | None
    embedding: np.ndarray | None
    aligned: np.ndarray | None  # 112x112 crop, only on frames where the face was re-embedded
    fresh: bool
    mesh: MeshFace | None
    age: float
    hits: int
    embed_count: int

    @property
    def face_px(self) -> float:
        return float(min(self.bbox[2], self.bbox[3]))


@dataclass
class FrameResult:
    index: int
    t: float
    size: tuple[int, int]
    faces: list[FaceResult]
    timings: dict[str, float] = field(default_factory=dict)
    gallery_size: int = 0

    @property
    def primary(self) -> FaceResult | None:
        return self.faces[0] if self.faces else None  # faces are sorted largest first


@dataclass
class StillFace:
    """Result of analysing one face in a still image (no tracking)."""

    face: Face
    aligned: np.ndarray
    embedding: np.ndarray
    quality: fq.FaceQuality
    match: MatchResult


class FaceEngine:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: Matcher | None = None,
        ema_alpha: float = 0.5,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher or Matcher()
        self.tracker = FaceTracker()
        self.ema_alpha = ema_alpha
        self._gallery = Gallery([], [])
        self._mesh: MeshAnalyzer | None = None
        self._mesh_failed = False
        self._index = 0

    # ------------------------------------------------------------ construction
    @classmethod
    def create(cls, settings: Settings | None = None) -> "FaceEngine":
        settings = settings or Settings()
        paths = models.ensure_models(models.REQUIRED)
        detector = FaceDetector(
            paths["detector"], score_threshold=settings.detection_score, min_face_size=settings.min_face_size
        )
        embedder = FaceEmbedder(paths["recognizer"])
        engine = cls(detector, embedder)
        engine.configure(settings)
        return engine

    def configure(self, settings: Settings) -> None:
        self.matcher = Matcher(settings.match_threshold, settings.unknown_threshold, settings.top_k)
        self.detector.set_score_threshold(settings.detection_score)
        self.detector.min_face_size = settings.min_face_size
        for tr in self.tracker.tracks:  # re-decide with the new thresholds
            if tr.score_ema is not None:
                tr.match = self.matcher.decide(self._gallery, tr.score_ema)

    def set_gallery(self, gallery: Gallery) -> None:
        self._gallery = gallery

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    @property
    def mesh_available(self) -> bool:
        return self.ensure_mesh()

    def ensure_mesh(self) -> bool:
        if self._mesh is not None:
            return True
        if self._mesh_failed:
            return False
        try:
            if not mediapipe_available():
                raise RuntimeError("mediapipe is not installed")
            path = models.ensure_models(["landmarker"])["landmarker"]
            self._mesh = MeshAnalyzer(path, max_faces=4)
        except Exception as exc:
            log.warning("Face mesh unavailable: %s", exc)
            self._mesh_failed = True
            return False
        return True

    def reset(self) -> None:
        self.tracker.reset()

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    # ------------------------------------------------------------ per frame
    def _score(self, track: Track) -> None:
        gallery = self._gallery
        scores = gallery.scores(track.embedding, self.matcher.top_k)
        if track.score_ema is None or track.ema_version != gallery.version or track.score_ema.shape != scores.shape:
            track.score_ema = scores
            track.ema_version = gallery.version
        else:
            track.score_ema = self.ema_alpha * scores + (1.0 - self.ema_alpha) * track.score_ema
        track.match = self.matcher.decide(gallery, track.score_ema)

    def process(self, frame: np.ndarray, t: float | None = None, options: EngineOptions | None = None) -> FrameResult:
        options = options or EngineOptions()
        t = time.monotonic() if t is None else t
        self._index += 1
        t0 = time.perf_counter()
        faces = self.detector.detect(frame)
        t1 = time.perf_counter()
        tracks = self.tracker.update(faces, t)

        fresh: set[int] = set()
        if options.recognize:
            for tr in tracks:
                if len(fresh) >= options.max_embeds_per_frame:
                    break
                if tr.embedding is not None and t - tr.embedded_at < options.embed_interval:
                    continue
                aligned, emb = self.embedder.embed(frame, tr.face)
                tr.embedding, tr.aligned, tr.embedded_at = emb, aligned, t
                tr.embed_count += 1
                tr.quality = fq.assess(frame.shape, tr.face, aligned, options.quality)
                self._score(tr)
                fresh.add(tr.id)
            for tr in tracks:  # gallery changed since this track was last scored
                if tr.id not in fresh and tr.embedding is not None and tr.ema_version != self._gallery.version:
                    tr.score_ema = None
                    self._score(tr)
        t2 = time.perf_counter()

        if options.mesh and tracks and self.ensure_mesh():
            meshes = self._mesh.process(frame, t)
            for tr in tracks:
                tr.mesh = None
            for m in meshes:
                best = max(tracks, key=lambda tr: iou(tr.face.bbox, m.bbox))
                if iou(best.face.bbox, m.bbox) >= 0.3 and best.mesh is None:
                    best.mesh = m
        else:
            for tr in tracks:
                tr.mesh = None
        t3 = time.perf_counter()

        results = [
            FaceResult(
                track_id=tr.id,
                bbox=tr.face.bbox.copy(),
                landmarks=tr.face.landmarks.copy(),
                det_score=tr.face.score,
                pose=tr.face.pose,
                match=tr.match,
                quality=tr.quality,
                embedding=tr.embedding,
                aligned=tr.aligned if tr.id in fresh else None,
                fresh=tr.id in fresh,
                mesh=tr.mesh,
                age=tr.age,
                hits=tr.hits,
                embed_count=tr.embed_count,
            )
            for tr in tracks
        ]
        timings = {
            "detect_ms": (t1 - t0) * 1e3,
            "embed_ms": (t2 - t1) * 1e3,
            "mesh_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3,
            "embedded": float(len(fresh)),
        }
        return FrameResult(self._index, t, (frame.shape[1], frame.shape[0]), results, timings, len(self._gallery))

    # ------------------------------------------------------------ still images / tasks
    def embed_face(self, frame: np.ndarray, face: Face, tta: bool = True,
                   thresholds: fq.QualityThresholds = fq.ENROLL_THRESHOLDS) -> tuple[np.ndarray, np.ndarray, fq.FaceQuality]:
        """High-quality embedding (flip test-time augmentation) for enrollment and verification."""
        aligned = self.embedder.align(frame, face)
        emb = self.embedder.embed_aligned(aligned, tta=tta)
        return aligned, emb, fq.assess(frame.shape, face, aligned, thresholds)

    def analyze_image(self, image: np.ndarray, max_faces: int = 20) -> list[StillFace]:
        out = []
        for face in self.detector.detect(image)[:max_faces]:
            aligned, emb, q = self.embed_face(image, face, tta=True, thresholds=fq.PASSIVE_THRESHOLDS)
            out.append(StillFace(face, aligned, emb, q, self.matcher.identify(self._gallery, emb)))
        return out
