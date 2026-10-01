"""Face alignment + deep embedding with SFace (OpenCV ``FaceRecognizerSF``)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from faceguard.vision.detector import Face

EMBEDDING_DIM = 128
ALIGNED_SIZE = 112


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class FaceEmbedder:
    """Aligns a face to the canonical 112x112 template and maps it to a unit vector.

    Two faces of the same person have a high cosine similarity (dot product of
    the unit vectors); different people score low.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._net = cv2.FaceRecognizerSF.create(str(model_path), "")

    def align(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """Similarity-transform the face so the 5 landmarks land on the template (112x112 BGR)."""
        return self._net.alignCrop(frame, face.row)

    def embed_aligned(self, aligned: np.ndarray, tta: bool = False) -> np.ndarray:
        """128-d L2-normalised embedding. ``tta`` also embeds the mirror image and averages."""
        feat = self._net.feature(aligned).reshape(-1).astype(np.float32)
        feat = normalize(feat)
        if tta:
            flipped = self._net.feature(cv2.flip(aligned, 1)).reshape(-1).astype(np.float32)
            feat = normalize(feat + normalize(flipped))
        return feat

    def embed(self, frame: np.ndarray, face: Face, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(aligned_crop, embedding)``."""
        aligned = self.align(frame, face)
        return aligned, self.embed_aligned(aligned, tta=tta)
