from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from faceguard import models
from faceguard.recognition.engine import FaceResult, FrameResult
from faceguard.recognition.gallery import PENDING, MatchResult
from faceguard.vision.geometry import HeadPose
from faceguard.vision.mesh import MeshFace
from faceguard.vision.quality import FaceQuality

DATA = Path(__file__).parent / "data"
MODELS_OK = models.is_present("detector") and models.is_present("recognizer")
needs_models = pytest.mark.skipif(not MODELS_OK, reason="models not downloaded (python -m faceguard download-models)")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own empty data folder."""
    home = tmp_path / "home"
    monkeypatch.setenv("FACEGUARD_HOME", str(home))
    return home


# A simple 3-D face (mm; z < 0 is towards the camera): eyes, nose tip 36 mm in front, mouth corners.
FACE3D = np.array([[-31.5, -30, 0], [31.5, -30, 0], [0, 5, -36], [-25, 35, -8], [25, 35, -8]], dtype=float)
FLAT3D = FACE3D * np.array([1.0, 1.0, 0.0])  # the same face printed on a card


def render(points3d: np.ndarray, yaw_deg: float = 0.0, dist: float = 500.0, f: float = 457.0) -> np.ndarray:
    """Rotate about the vertical axis (+ = subject's left) and project with a pinhole webcam -> (5, 2) landmarks."""
    a = math.radians(yaw_deg)
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    xr, zr = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
    return np.stack([f * xr / (dist + zr) + 320, f * y / (dist + zr) + 240], axis=1)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def identity(rng: np.random.Generator, dim: int = 128) -> np.ndarray:
    return unit(rng.normal(size=dim))


def samples_of(center: np.ndarray, n: int, rng: np.random.Generator, spread: float = 0.066) -> np.ndarray:
    """Noisy unit vectors with cosine ~0.8 to ``center`` (like real embeddings of one person)."""
    return unit(center + spread * rng.normal(size=(n, center.size)))


def ok_quality(yaw: float = 0.0, pitch: float = 0.0, issues: tuple[str, ...] = ()) -> FaceQuality:
    return FaceQuality(200.0, 120.0, 40.0, 0.0, 150.0, 0.95, yaw, pitch, True, issues, 0.9)


def face_result(track_id: int = 1, yaw: float = 0.0, pitch: float = 0.0, embedding: np.ndarray | None = None,
                fresh: bool = False, match: MatchResult = PENDING, blink: float | None = None,
                smile: float | None = None, size: float = 150.0, x: float = 200.0, embed_count: int = 3,
                quality: FaceQuality | None = None) -> FaceResult:
    mesh = None
    if blink is not None or smile is not None:
        mesh = MeshFace(np.zeros((478, 2), np.float32), np.array([x, 100, size, size], np.float32),
                        blink or 0.0, blink or 0.0, smile or 0.0, 0.0, yaw, pitch, 0.0)
    return FaceResult(
        track_id=track_id, bbox=np.array([x, 100, size, size], np.float32),
        landmarks=render(FACE3D, yaw).astype(np.float32),  # consistent with the yaw, like a real face
        det_score=0.95, pose=HeadPose(yaw, pitch, 0.0), match=match, quality=quality or ok_quality(yaw, pitch),
        embedding=embedding, aligned=np.zeros((112, 112, 3), np.uint8) if fresh else None, fresh=fresh, mesh=mesh,
        age=1.0, hits=10, embed_count=embed_count,
    )


def frame_result(t: float, faces: list[FaceResult], index: int = 0) -> FrameResult:
    return FrameResult(index, t, (1280, 720), faces, {}, 0)
