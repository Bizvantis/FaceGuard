"""Dense face mesh with MediaPipe Face Landmarker (optional dependency).

Provides 478 3-D landmarks, 52 ARKit-style blendshape scores (used for the
blink and smile liveness challenges) and a 4x4 facial transformation matrix
from which the head pose is derived.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from faceguard.vision.geometry import bbox_of_points

log = logging.getLogger(__name__)


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class MeshFace:
    points: np.ndarray  # (478, 2) float32, pixel coordinates
    bbox: np.ndarray  # [x, y, w, h]
    blink_left: float
    blink_right: float
    smile: float
    jaw_open: float
    yaw: float  # degrees, same convention as geometry.HeadPose
    pitch: float
    roll: float

    @property
    def blink(self) -> float:
        return (self.blink_left + self.blink_right) / 2.0


_TESSELATION: np.ndarray | None = None
_CONTOURS: np.ndarray | None = None


def _edges(name: str) -> np.ndarray:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    conns = getattr(C, name)
    return np.array([(c.start, c.end) for c in conns], dtype=np.int32)


def tesselation_edges() -> np.ndarray:
    global _TESSELATION
    if _TESSELATION is None:
        _TESSELATION = _edges("FACE_LANDMARKS_TESSELATION")
    return _TESSELATION


def contour_edges() -> np.ndarray:
    global _CONTOURS
    if _CONTOURS is None:
        _CONTOURS = _edges("FACE_LANDMARKS_CONTOURS")
    return _CONTOURS


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in FaceGuard's conventions from MediaPipe's transformation matrix."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    pitch_down = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, -pitch_down, roll


class MeshAnalyzer:
    """Runs MediaPipe Face Landmarker in VIDEO mode (uses temporal tracking between frames)."""

    def __init__(self, model_path: str | Path, max_faces: int = 2) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            # Load from memory so unusual characters in the install path cannot break the native loader.
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t_seconds: float) -> list[MeshFace]:
        h, w = frame_bgr.shape[:2]
        ts = max(int(t_seconds * 1000.0), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        faces: list[MeshFace] = []
        for i, landmarks in enumerate(result.face_landmarks):
            pts = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)
            scores = {}
            if result.face_blendshapes and i < len(result.face_blendshapes):
                scores = {c.category_name: float(c.score) for c in result.face_blendshapes[i]}
            yaw = pitch = roll = 0.0
            if result.facial_transformation_matrixes and i < len(result.facial_transformation_matrixes):
                yaw, pitch, roll = euler_from_matrix(result.facial_transformation_matrixes[i])
            faces.append(
                MeshFace(
                    points=pts,
                    bbox=bbox_of_points(pts),
                    blink_left=scores.get("eyeBlinkLeft", 0.0),
                    blink_right=scores.get("eyeBlinkRight", 0.0),
                    smile=(scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)) / 2.0,
                    jaw_open=scores.get("jawOpen", 0.0),
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )
        return faces

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - best effort
            pass
