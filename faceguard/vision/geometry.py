"""Geometry helpers: head pose from the 5 YuNet landmarks, box overlap.

Landmark order (YuNet): right eye, left eye, nose tip, right mouth corner,
left mouth corner - "right"/"left" from the subject's point of view, so in a
normal (non-mirrored) camera image the right eye appears on the image's left.

Sign conventions used everywhere in FaceGuard:

* ``yaw``   > 0  -> subject turned their head to *their* left
  (the nose moves towards the image's right edge in the raw camera frame)
* ``pitch`` > 0  -> subject is looking up
* ``roll``  angle of the eye line in the image, degrees

The constants below were calibrated on real 3-D face geometry (MediaPipe meshes of
300 LFW faces rendered through a webcam model, ``experiments/exp_parallax.py``) so
that the estimates are in physical degrees; on 2,998 real LFW photos the yaw agrees
with MediaPipe's 3-D head pose with r = 0.92 (``experiments/exp_pose.py``).

The yaw *ratio* alone can be fooled by a printed photo tilted steeply towards the
camera (perspective tilts the eye line, and undoing that roll leaks the nose's
vertical offset into the horizontal).  ``parallax_yaw`` closes that gap: it is
exactly zero for anything planar, however it moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# nose offset / inter-ocular distance ~= K * tan(yaw)      (3-D calibration: K = 0.689)
_YAW_K = 0.690
# pitch_down_deg ~= SLOPE * (ratio - 0.52), ratio = nose height between eye line (0) and mouth line (1)
_PITCH_SLOPE = 90.0
NEUTRAL_PITCH_RATIO = 0.52


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    yaw_ratio: float = 0.0
    pitch_ratio: float = NEUTRAL_PITCH_RATIO

    def is_frontal(self, max_yaw: float = 15.0, max_pitch: float = 20.0) -> bool:
        return abs(self.yaw) <= max_yaw and abs(self.pitch) <= max_pitch


def pose_from_landmarks(landmarks: np.ndarray) -> HeadPose:
    """Approximate head pose (degrees) from the 5 facial landmarks."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    right_eye, left_eye, nose, mouth_r, mouth_l = lm
    eye_vec = left_eye - right_eye
    iod = float(math.hypot(eye_vec[0], eye_vec[1]))
    if iod < 1e-6:
        return HeadPose(0.0, 0.0, 0.0)
    roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

    # Undo the in-plane rotation so that the eye line is horizontal.
    c, s = eye_vec / iod
    derot = np.array([[c, s], [-s, c]])
    eye_mid = (right_eye + left_eye) / 2.0
    nose_r = derot @ (nose - eye_mid)
    mouth_r_ = derot @ ((mouth_r + mouth_l) / 2.0 - eye_mid)

    yaw_ratio = float(nose_r[0] / iod)
    yaw = math.degrees(math.atan(yaw_ratio / _YAW_K))

    depth = float(mouth_r_[1])
    pitch_ratio = float(nose_r[1] / depth) if depth > 1e-6 else NEUTRAL_PITCH_RATIO
    pitch = -_PITCH_SLOPE * (pitch_ratio - NEUTRAL_PITCH_RATIO)
    pitch = max(-60.0, min(60.0, pitch))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll, yaw_ratio=yaw_ratio, pitch_ratio=pitch_ratio)


_PLANE_POINTS = [0, 1, 3, 4]  # eyes and mouth corners: nearly coplanar on a real face
_PARALLAX_K = 0.628  # parallax ratio ~= K * tan(yaw) for a real head (calibrated in experiments/exp_parallax.py)


def parallax_ratio(base: np.ndarray, current: np.ndarray) -> float:
    """Out-of-plane motion of the nose, as a fraction of the inter-ocular distance.

    The homography that maps the baseline eye / mouth-corner points onto the current ones describes *any*
    motion of a flat object - translation, scale, roll, tilt, even strong perspective.  A printed photo or a
    screen therefore keeps its nose exactly where that homography predicts; only a real head, whose nose
    sticks out of the eye/mouth plane, moves it away.  Positive = subject turned to their left.
    """
    import cv2

    b = np.asarray(base, dtype=np.float32).reshape(5, 2)
    c = np.asarray(current, dtype=np.float32).reshape(5, 2)
    try:
        h = cv2.getPerspectiveTransform(b[_PLANE_POINTS], c[_PLANE_POINTS])
    except cv2.error:
        return 0.0
    p = h @ np.array([b[2, 0], b[2, 1], 1.0])
    if abs(p[2]) < 1e-9:
        return 0.0
    residual = c[2].astype(np.float64) - p[:2] / p[2]
    eye = (c[1] - c[0]).astype(np.float64)
    iod = math.hypot(eye[0], eye[1])
    if iod < 1e-6:
        return 0.0
    return float(residual @ (eye / iod) / iod)


def parallax_yaw(base: np.ndarray, current: np.ndarray) -> float:
    """Head yaw since ``base`` (degrees) measured purely from nose parallax; ~0 for any planar object."""
    return math.degrees(math.atan(parallax_ratio(base, current) / _PARALLAX_K))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_of_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def clamp_box(box: np.ndarray, width: int, height: int, margin: float = 0.0) -> tuple[int, int, int, int]:
    """Expand a box by ``margin`` (fraction of its size) and clip it to the image; returns x0, y0, x1, y1."""
    x, y, w, h = (float(v) for v in box[:4])
    x -= w * margin
    y -= h * margin
    w *= 1 + 2 * margin
    h *= 1 + 2 * margin
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
    return x0, y0, max(x0, x1), max(y0, y1)
