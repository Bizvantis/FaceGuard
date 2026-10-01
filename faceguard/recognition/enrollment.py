"""Guided face enrollment.

The user is walked through several head poses (straight, left, right, up,
down).  A sample is only captured when the pose is right *and* the image passes
the quality gate, so the stored template covers the natural range of
viewpoints and lighting instead of 20 copies of the same frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from faceguard.recognition.database import SOURCE_CAMERA, SOURCE_PHOTO, Sample
from faceguard.recognition.engine import FaceEngine, FrameResult
from faceguard.recognition.gallery import Gallery, Matcher, MatchResult, PersonInfo
from faceguard.vision import quality as fq
from faceguard.vision.camera import load_image
from faceguard.vision.detector import Face
from faceguard.vision.embedder import normalize


@dataclass(frozen=True)
class PoseTarget:
    key: str
    label: str
    instruction: str
    count: int
    yaw: tuple[float, float]  # degrees; + = the user's left
    pitch: tuple[float, float]  # degrees relative to the user's own neutral pitch; + = up


DEFAULT_TARGETS: tuple[PoseTarget, ...] = (
    PoseTarget("center", "Straight", "Look straight at the camera", 8, (-10.0, 10.0), (-8.0, 8.0)),
    PoseTarget("left", "Left", "Slowly turn your head to the LEFT", 4, (12.0, 40.0), (-15.0, 15.0)),
    PoseTarget("right", "Right", "Slowly turn your head to the RIGHT", 4, (-40.0, -12.0), (-15.0, 15.0)),
    PoseTarget("up", "Up", "Tilt your chin UP a little", 3, (-12.0, 12.0), (5.0, 30.0)),
    PoseTarget("down", "Down", "Tilt your chin DOWN a little", 3, (-12.0, 12.0), (-30.0, -5.0)),
)


@dataclass
class TargetProgress:
    target: PoseTarget
    collected: int = 0
    status: str = "pending"  # pending | active | done | skipped


@dataclass
class EnrollmentState:
    phase: str  # running | done | cancelled
    instruction: str
    hint: str
    progress: float
    targets: list[TargetProgress]
    current: PoseTarget | None
    captured: bool  # a sample was captured on this frame
    face_bbox: np.ndarray | None
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass
class EnrollmentResult:
    name: str
    samples: list[Sample]
    thumbnails: list[np.ndarray]
    consistency: float  # mean cosine similarity of samples to their centroid
    skipped: list[str] = field(default_factory=list)
    rejected: int = 0

    @property
    def centroid(self) -> np.ndarray:
        return normalize(np.mean([s.embedding for s in self.samples], axis=0))


def _face_of(result_face) -> Face:
    return Face(bbox=result_face.bbox, landmarks=result_face.landmarks, score=result_face.det_score)


class EnrollmentSession:
    """Feed it every processed frame (in the camera thread) until ``state.phase == 'done'``."""

    def __init__(
        self,
        name: str,
        targets: Iterable[PoseTarget] = DEFAULT_TARGETS,
        min_interval: float = 0.25,
        target_timeout: float = 12.0,
        thresholds: fq.QualityThresholds = fq.ENROLL_THRESHOLDS,
    ) -> None:
        self.name = name
        self.progress = [TargetProgress(t) for t in targets]
        self.min_interval = min_interval
        self.target_timeout = target_timeout
        self.thresholds = thresholds
        self._index = 0
        self._samples: list[Sample] = []
        self._aligned: list[tuple[str, np.ndarray]] = []
        self._by_target: dict[str, list[np.ndarray]] = {t.target.key: [] for t in self.progress}
        self._last_capture = -1e9
        self._target_started: float | None = None
        self._pitch_samples: list[float] = []
        self._rejected = 0
        self.phase = "running"
        self.progress[0].status = "active"

    # ------------------------------------------------------------ helpers
    @property
    def current(self) -> PoseTarget | None:
        return self.progress[self._index].target if self._index < len(self.progress) else None

    @property
    def baseline_pitch(self) -> float | None:
        return float(np.median(self._pitch_samples)) if len(self._pitch_samples) >= 3 else None

    def _fraction(self) -> float:
        needed = sum(p.target.count for p in self.progress if p.status != "skipped")
        got = sum(min(p.collected, p.target.count) for p in self.progress if p.status != "skipped")
        return got / needed if needed else 1.0

    def _advance(self, t: float, skipped: bool = False) -> None:
        self.progress[self._index].status = "skipped" if skipped else "done"
        self._index += 1
        self._target_started = t
        if self._index >= len(self.progress):
            self.phase = "done"
        else:
            self.progress[self._index].status = "active"

    def skip_current(self, t: float | None = None) -> None:
        if self.phase == "running" and self.current and self.current.key != "center":
            self._advance(time.monotonic() if t is None else t, skipped=True)

    def cancel(self) -> None:
        self.phase = "cancelled"

    def _state(self, hint: str, captured: bool, bbox: np.ndarray | None, yaw: float = 0.0, pitch: float = 0.0) -> EnrollmentState:
        cur = self.current
        instruction = cur.instruction if cur else "All done!"
        return EnrollmentState(self.phase, instruction, hint, self._fraction(), self.progress, cur, captured, bbox, yaw, pitch)

    # ------------------------------------------------------------ main update
    def update(self, frame: np.ndarray, result: FrameResult, engine: FaceEngine, t: float | None = None) -> EnrollmentState:
        t = time.monotonic() if t is None else t
        if self.phase != "running":
            return self._state("", False, None)
        if self._target_started is None:
            self._target_started = t
        target = self.current
        assert target is not None

        if target.key != "center" and t - self._target_started > self.target_timeout:
            self._advance(t, skipped=True)
            return self._state("Skipped that pose - no problem", False, None)

        faces = result.faces
        if not faces:
            return self._state("No face detected - look at the camera", False, None)
        primary = faces[0]
        if len(faces) > 1 and faces[1].bbox[2] * faces[1].bbox[3] > 0.25 * primary.bbox[2] * primary.bbox[3]:
            return self._state("Only one person should be in view", False, primary.bbox)

        pose = primary.pose
        rel_pitch = pose.pitch - (self.baseline_pitch or 0.0)
        pitch_for_target = rel_pitch if (target.key != "center" or self.baseline_pitch is not None) else 0.0
        in_pose = target.yaw[0] <= pose.yaw <= target.yaw[1] and target.pitch[0] <= pitch_for_target <= target.pitch[1]
        if not in_pose:
            return self._state(self._direction_hint(target, pose.yaw, pitch_for_target), False, primary.bbox, pose.yaw, rel_pitch)
        if t - self._last_capture < self.min_interval:
            return self._state("Hold that pose...", False, primary.bbox, pose.yaw, rel_pitch)

        face = _face_of(primary)
        aligned, emb, q = engine.embed_face(frame, face, tta=True, thresholds=self.thresholds)
        if not q.ok:
            return self._state(q.issues[0], False, primary.bbox, pose.yaw, rel_pitch)
        same_target = self._by_target[target.key]
        if same_target and max(float(e @ emb) for e in same_target) > 0.995:
            return self._state("Move a tiny bit...", False, primary.bbox, pose.yaw, rel_pitch)
        if len(self._samples) >= 3:
            centroid = normalize(np.mean([s.embedding for s in self._samples], axis=0))
            if float(centroid @ emb) < 0.35:
                self._rejected += 1
                return self._state("That looks like a different face - only you should be in view", False, primary.bbox)

        self._samples.append(Sample(emb, SOURCE_CAMERA, q.score, pose.yaw, rel_pitch, time.time()))
        self._aligned.append((target.key, aligned))
        same_target.append(emb)
        self._last_capture = t
        if target.key == "center":
            self._pitch_samples.append(pose.pitch)
        prog = self.progress[self._index]
        prog.collected += 1
        if prog.collected >= target.count:
            self._advance(t)
        return self._state("Captured", True, primary.bbox, pose.yaw, rel_pitch)

    @staticmethod
    def _direction_hint(target: PoseTarget, yaw: float, pitch: float) -> str:
        if yaw < target.yaw[0]:
            return "Turn a bit more to your LEFT" if target.yaw[0] > 0 else "Turn back towards the camera"
        if yaw > target.yaw[1]:
            return "Turn a bit more to your RIGHT" if target.yaw[1] < 0 else "Turn back towards the camera"
        if pitch < target.pitch[0]:
            return "Chin up a little" if target.pitch[0] > 0 else "Level your head"
        if pitch > target.pitch[1]:
            return "Chin down a little" if target.pitch[1] < 0 else "Level your head"
        return target.instruction

    # ------------------------------------------------------------ result
    def finish(self) -> EnrollmentResult:
        samples = list(self._samples)
        if not samples:
            raise ValueError("No samples were captured")
        embs = np.stack([s.embedding for s in samples])
        centroid = normalize(embs.mean(axis=0))
        sims = embs @ centroid
        keep = sims >= 0.40  # drop outliers (bad alignment, someone walking past)
        if keep.sum() >= max(3, len(samples) // 2):
            samples = [s for s, k in zip(samples, keep) if k]
            sims = sims[keep]
        thumbs: list[np.ndarray] = []
        for key in ("center", "left", "right"):
            crop = next((a for k, a in self._aligned if k == key), None)
            if crop is not None:
                thumbs.append(crop)
        skipped = [p.target.label for p in self.progress if p.status == "skipped"]
        return EnrollmentResult(self.name, samples, thumbs, float(sims.mean()), skipped, self._rejected)


def find_duplicate(gallery: Gallery, matcher: Matcher, centroid: np.ndarray, exclude_id: str | None = None) -> MatchResult | None:
    """If the new face already matches someone else in the gallery, return that match."""
    if gallery.empty:
        return None
    scores = gallery.scores(centroid, matcher.top_k)
    for j in np.argsort(scores)[::-1]:
        person: PersonInfo = gallery.persons[int(j)]
        if person.id == exclude_id:
            continue
        if scores[j] >= matcher.threshold:
            return MatchResult("match", person, float(scores[j]))
        break
    return None


@dataclass
class PhotoImportReport:
    samples: list[Sample]
    thumbnails: list[np.ndarray]
    used: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)


def samples_from_photos(engine: FaceEngine, files: Iterable[str | Path]) -> PhotoImportReport:
    """Enroll from existing photos: largest face per photo, with a relaxed quality gate."""
    report = PhotoImportReport([], [])
    accepted: list[tuple[str, Sample, np.ndarray]] = []
    for f in files:
        name = Path(f).name
        img = load_image(f)
        if img is None:
            report.skipped.append((name, "cannot read image"))
            continue
        faces = engine.detector.detect(img)
        if not faces:
            report.skipped.append((name, "no face found"))
            continue
        face = faces[0]
        if len(faces) > 1 and faces[1].area > 0.5 * face.area:
            report.skipped.append((name, "several faces of similar size - crop the photo to one person"))
            continue
        aligned, emb, q = engine.embed_face(img, face, tta=True, thresholds=fq.PASSIVE_THRESHOLDS)
        if q.face_px < 40 or q.sharpness < 15:
            report.skipped.append((name, "face too small or blurry"))
            continue
        accepted.append((name, Sample(emb, SOURCE_PHOTO, q.score, q.yaw, q.pitch, time.time()), aligned))
    if len(accepted) >= 3:  # drop photos that disagree with the majority (probably a different person)
        embs = np.stack([s.embedding for _, s, _ in accepted])
        keep = embs @ normalize(embs.mean(axis=0)) >= 0.35
        for (name, _, _), k in zip(accepted, keep):
            if not k:
                report.skipped.append((name, "does not look like the same person as the other photos"))
        accepted = [a for a, k in zip(accepted, keep) if k]
    report.used = [name for name, _, _ in accepted]
    report.samples = [s for _, s, _ in accepted]
    report.thumbnails = [a for _, _, a in accepted[:3]]
    return report
