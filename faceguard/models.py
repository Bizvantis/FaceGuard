"""Registry and downloader for the pretrained neural networks.

All three models are downloaded from their official publishers and verified
against pinned SHA-256 hashes before use.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from faceguard import paths

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # (model key, bytes done, bytes total)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    sha256: str
    size: int
    urls: tuple[str, ...]
    title: str
    source: str
    license: str

    @property
    def path(self) -> Path:
        return paths.models_dir() / self.filename


MODELS: dict[str, ModelSpec] = {
    "detector": ModelSpec(
        key="detector",
        filename="face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232_589,
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        title="YuNet face detector (CNN, 5 landmarks)",
        source="OpenCV Model Zoo",
        license="MIT",
    ),
    "recognizer": ModelSpec(
        key="recognizer",
        filename="face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size=38_696_353,
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
        title="SFace face recognizer (MobileFaceNet, 128-d embedding)",
        source="OpenCV Model Zoo",
        license="Apache-2.0",
    ),
    "landmarker": ModelSpec(
        key="landmarker",
        filename="face_landmarker.task",
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        size=3_758_596,
        urls=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        ),
        title="MediaPipe Face Landmarker (478 points + 52 blendshapes)",
        source="Google MediaPipe",
        license="Apache-2.0",
    ),
}

REQUIRED = ("detector", "recognizer")


class ModelError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_present(key: str, verify: bool = False) -> bool:
    spec = MODELS[key]
    p = spec.path
    if not p.is_file() or p.stat().st_size != spec.size:
        return False
    return not verify or sha256_of(p) == spec.sha256


def _download(spec: ModelSpec, progress: ProgressFn | None) -> Path:
    dest = spec.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in spec.urls:
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{spec.filename}.", suffix=".part")
        tmp = Path(tmp_name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FaceGuard/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, os.fdopen(fd, "wb") as out:
                total = int(resp.headers.get("Content-Length") or spec.size)
                done = 0
                h = hashlib.sha256()
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(spec.key, done, total)
            if h.hexdigest() != spec.sha256:
                raise ModelError(f"checksum mismatch for {spec.filename} from {url}")
            os.replace(tmp, dest)
            log.info("Downloaded %s (%d bytes)", spec.filename, done)
            return dest
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
            log.warning("Download failed: %s", errors[-1])
        finally:
            tmp.unlink(missing_ok=True)
    raise ModelError(f"Could not download {spec.filename}:\n  " + "\n  ".join(errors))


def ensure_models(keys: Iterable[str] = REQUIRED, progress: ProgressFn | None = None) -> dict[str, Path]:
    """Return paths to the requested models, downloading any that are missing."""
    out: dict[str, Path] = {}
    for key in keys:
        spec = MODELS[key]
        if not is_present(key):
            _download(spec, progress)
        out[key] = spec.path
    return out


def model_path(key: str) -> Path:
    """Path to an already-present model (raises ModelError if it is missing)."""
    if not is_present(key):
        spec = MODELS[key]
        raise ModelError(
            f"Model '{spec.filename}' is missing. Run:  python -m faceguard download-models"
        )
    return MODELS[key].path
