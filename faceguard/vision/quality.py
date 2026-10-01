"""Face image quality assessment.

Recognition accuracy depends heavily on the input.  These checks were
calibrated on real data (see README, "Calibration"): SFace similarity survives
dim light well but collapses with over-exposure, blur below a Laplacian
variance of ~30, and faces narrower than ~40 px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from faceguard.vision.detector import Face


@dataclass(frozen=True)
class QualityThresholds:
    min_face_px: float = 80.0
    min_sharpness: float = 35.0
    min_brightness: float = 45.0
    max_brightness: float = 200.0
    max_overexposed: float = 0.12
    min_det_score: float = 0.80
    max_yaw: float = 40.0
    max_pitch: float = 35.0
    frame_margin: float = 2.0


ENROLL_THRESHOLDS = QualityThresholds()
# Looser gate used for passive decisions (intruder / adaptive learning) at a distance.
PASSIVE_THRESHOLDS = QualityThresholds(min_face_px=60.0, min_sharpness=25.0, max_yaw=30.0, max_pitch=30.0)


@dataclass(frozen=True)
class FaceQuality:
    sharpness: float
    brightness: float
    contrast: float
    overexposed: float
    face_px: float
    det_score: float
    yaw: float
    pitch: float
    in_frame: bool
    issues: tuple[str, ...]
    score: float

    @property
    def ok(self) -> bool:
        return not self.issues


def image_stats(aligned: np.ndarray) -> tuple[float, float, float, float]:
    """(sharpness, brightness, contrast, overexposed fraction) of an aligned 112x112 face."""
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    core = gray[int(h * 0.2): int(h * 0.8), int(w * 0.2): int(w * 0.8)]
    return sharpness, float(core.mean()), float(core.std()), float(np.mean(core >= 250))


def assess(
    frame_shape: tuple[int, ...],
    face: Face,
    aligned: np.ndarray,
    thresholds: QualityThresholds = ENROLL_THRESHOLDS,
) -> FaceQuality:
    sharpness, brightness, contrast, overexposed = image_stats(aligned)
    fh, fw = frame_shape[:2]
    x, y, w, h = (float(v) for v in face.bbox)
    m = thresholds.frame_margin
    in_frame = x >= m and y >= m and x + w <= fw - m and y + h <= fh - m
    pose = face.pose
    face_px = min(w, h)

    issues: list[str] = []
    if not in_frame:
        issues.append("Keep your whole face inside the frame")
    if face_px < thresholds.min_face_px:
        issues.append("Move closer to the camera")
    if brightness < thresholds.min_brightness:
        issues.append("Too dark - add some light in front of you")
    elif brightness > thresholds.max_brightness or overexposed > thresholds.max_overexposed:
        issues.append("Too bright - avoid glare or light behind the camera")
    if sharpness < thresholds.min_sharpness:
        issues.append("Hold still - the image is blurry")
    if abs(pose.yaw) > thresholds.max_yaw or abs(pose.pitch) > thresholds.max_pitch:
        issues.append("Face the camera")
    if face.score < thresholds.min_det_score:
        issues.append("Face not clearly visible")

    size_q = min(1.0, max(0.0, (face_px - 40.0) / 80.0))
    sharp_q = min(1.0, max(0.0, math.log(max(sharpness, 1.0) / 10.0) / math.log(20.0)))
    light_q = 1.0 - min(1.0, abs(brightness - 125.0) / 100.0)
    expo_q = 1.0 - min(1.0, overexposed / 0.3)
    pose_q = 1.0 - min(1.0, max(abs(pose.yaw) / 45.0, abs(pose.pitch) / 40.0))
    score = 0.25 * size_q + 0.25 * sharp_q + 0.15 * light_q + 0.1 * expo_q + 0.15 * pose_q + 0.1 * face.score
    return FaceQuality(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        overexposed=overexposed,
        face_px=face_px,
        det_score=face.score,
        yaw=pose.yaw,
        pitch=pose.pitch,
        in_frame=in_frame,
        issues=tuple(issues),
        score=float(score),
    )
