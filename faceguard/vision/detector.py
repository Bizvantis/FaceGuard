"""Face detection with the YuNet CNN (OpenCV ``FaceDetectorYN``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from faceguard.vision.geometry import HeadPose, pose_from_landmarks

try:  # OpenCV 5 warns that DNN "targets" are unsupported by its new graph engine; harmless.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # pragma: no cover - very old OpenCV builds
    pass


@dataclass
class Face:
    """One detected face, in full-resolution frame coordinates."""

    bbox: np.ndarray  # float32 [x, y, w, h]
    landmarks: np.ndarray  # float32 (5, 2)
    score: float
    _pose: HeadPose | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_row(cls, row: np.ndarray) -> "Face":
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        return cls(bbox=row[0:4].copy(), landmarks=row[4:14].reshape(5, 2).copy(), score=float(row[14]))

    @property
    def row(self) -> np.ndarray:
        """The 15-float layout OpenCV's ``FaceRecognizerSF.alignCrop`` expects."""
        return np.concatenate([self.bbox, self.landmarks.reshape(-1), [self.score]]).astype(np.float32)

    @property
    def width(self) -> float:
        return float(self.bbox[2])

    @property
    def height(self) -> float:
        return float(self.bbox[3])

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return float(x + w / 2), float(y + h / 2)

    @property
    def pose(self) -> HeadPose:
        if self._pose is None:
            self._pose = pose_from_landmarks(self.landmarks)
        return self._pose


class FaceDetector:
    """YuNet face detector.

    Frames wider than ``max_width`` are downscaled for detection (YuNet is
    accurate down to ~20 px faces) and the results are mapped back to
    full-resolution coordinates so alignment can use every available pixel.
    """

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        max_width: int = 640,
        min_face_size: float = 20.0,
    ) -> None:
        self.max_width = max_width
        self.min_face_size = min_face_size
        self._net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold, 5000)
        self._input_size: tuple[int, int] | None = None

    def set_score_threshold(self, value: float) -> None:
        self._net.setScoreThreshold(float(value))

    def detect(self, frame: np.ndarray) -> list[Face]:
        if frame is None or frame.size == 0:
            return []
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / float(w))
        img = frame if scale >= 1.0 else cv2.resize(
            frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
        size = (img.shape[1], img.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, out = self._net.detect(img)
        if out is None:
            return []
        faces: list[Face] = []
        for row in out:
            row = row.astype(np.float32).copy()
            if scale < 1.0:
                row[:14] /= scale
            face = Face.from_row(row)
            if min(face.width, face.height) >= self.min_face_size:
                faces.append(face)
        faces.sort(key=lambda f: -f.area)
        return faces
