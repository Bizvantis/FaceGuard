"""Passive RGB presentation-attack detectors used as baselines (ONNX Runtime, CPU).

Every model returns the probability that a frame shows a presentation attack (0 = bona fide, 1 = attack),
with the preprocessing of its authors:

* ``minifasnet``  Silent-Face-Anti-Spoofing (Minivision): MiniFASNetV2 (crop scale 2.7) + MiniFASNetV1SE (4.0),
  80x80 BGR crops without normalisation, softmax over (print, live, replay) averaged over the two models.
  ONNX export of yakhyo/face-anti-spoofing, Apache-2.0.
* ``mn3``         anti-spoof-mn3 (Intel Open Model Zoo; MobileNetV3 trained on CelebA-Spoof), face box widened
  by 1.10 x 1.05, 128x128 RGB, mean (151.24, 119.60, 107.84), scale (63.01, 56.46, 55.00); output (live, spoof).
* ``cvpr24_r50``  He et al., CVPR 2024 FAS workshop (arXiv:2404.08450), ResNet-50 "general scenario" model on the
  full frame resized to 224x224, RGB, ImageNet normalisation; output (live, spoof). MIT licence.
* ``cvpr24_swin`` the same authors' Swin-V2-B model for the face area (face box + 20 px), otherwise as above.

The two CVPR 2024 checkpoints are converted to ONNX by ``pad_convert.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PAD_DIR = Path(__file__).resolve().parent / "data" / "pad"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


def _session(path: Path, threads: int) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _as_prob(out: np.ndarray) -> np.ndarray:
    out = np.asarray(out, np.float64).reshape(-1)
    if out.min() >= 0 and abs(out.sum() - 1.0) < 1e-3:
        return out
    return _softmax(out)


def _scaled_crop(img: np.ndarray, bbox, scale: float, size: tuple[int, int]) -> np.ndarray:
    """Silent-Face crop: a box ``scale`` times the face box, clipped to the image (CropImage._get_new_box)."""
    h_img, w_img = img.shape[:2]
    x, y, w, h = (float(v) for v in bbox[:4])
    scale = min((h_img - 1) / max(h, 1.0), (w_img - 1) / max(w, 1.0), scale)
    nw, nh = w * scale, h * scale
    cx, cy = x + w / 2.0, y + h / 2.0
    x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    x2, y2 = min(w_img - 1, int(cx + nw / 2)), min(h_img - 1, int(cy + nh / 2))
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], size)


def _box_crop(img: np.ndarray, bbox, ex: float, ey: float, pad_px: float = 0.0) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    x, y, w, h = (float(v) for v in bbox[:4])
    dx, dy = w * (ex - 1.0) / 2.0 + pad_px, h * (ey - 1.0) / 2.0 + pad_px
    x1, y1 = max(0, int(x - dx)), max(0, int(y - dy))
    x2, y2 = min(w_img, int(x + w + dx)), min(h_img, int(y + h + dy))
    return img[y1:y2, x1:x2]


@dataclass
class PadModel:
    name: str
    label: str
    files: tuple[str, ...]
    threads: int = 1

    def __post_init__(self):
        self.sessions = [_session(PAD_DIR / f, self.threads) for f in self.files]
        self.inputs = [s.get_inputs()[0].name for s in self.sessions]

    @property
    def available(self) -> bool:
        return all((PAD_DIR / f).exists() for f in self.files)

    def predict(self, frame_bgr: np.ndarray, bbox) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def timed(self, frame_bgr: np.ndarray, bbox) -> tuple[float, float]:
        t0 = time.perf_counter()
        p = self.predict(frame_bgr, bbox)
        return p, (time.perf_counter() - t0) * 1000.0


class MiniFAS(PadModel):
    def __init__(self, threads: int = 1):
        super().__init__("minifasnet", "MiniFASNet V2+V1SE", ("MiniFASNetV2.onnx", "MiniFASNetV1SE.onnx"), threads)

    def predict(self, frame_bgr, bbox) -> float:
        live = []
        for sess, name, scale in zip(self.sessions, self.inputs, (2.7, 4.0)):
            crop = _scaled_crop(frame_bgr, bbox, scale, (80, 80)).astype(np.float32)
            x = crop.transpose(2, 0, 1)[None]
            live.append(_as_prob(sess.run(None, {name: x})[0])[1])  # classes: print, live, replay
        return float(1.0 - np.mean(live))


class MN3(PadModel):
    MEAN = np.array([151.2405, 119.5950, 107.8395], np.float32)
    SCALE = np.array([63.0105, 56.4570, 55.0035], np.float32)

    def __init__(self, threads: int = 1):
        super().__init__("mn3", "anti-spoof-mn3", ("anti-spoof-mn3.onnx",), threads)

    def predict(self, frame_bgr, bbox) -> float:
        crop = _box_crop(frame_bgr, bbox, 1.10, 1.05)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((rgb - self.MEAN) / self.SCALE).transpose(2, 0, 1)[None]
        return float(_as_prob(self.sessions[0].run(None, {self.inputs[0]: x})[0])[1])


class CVPR24(PadModel):
    def __init__(self, name: str, label: str, file: str, face: bool, threads: int = 3):
        self.face = face
        super().__init__(name, label, (file,), threads)

    def predict(self, frame_bgr, bbox) -> float:
        img = _box_crop(frame_bgr, bbox, 1.0, 1.0, pad_px=20.0) if self.face else frame_bgr
        img = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        return float(_as_prob(self.sessions[0].run(None, {self.inputs[0]: x})[0])[1])


def light_models() -> list[PadModel]:
    return [MiniFAS(1), MN3(1)]


def heavy_models(threads: int = 3) -> list[PadModel]:
    out: list[PadModel] = []
    for name, label, file, face in (("cvpr24_r50", "CVPR24 ResNet-50 (full frame)", "full_resnet50.onnx", False),
                                    ("cvpr24_swin", "CVPR24 Swin-V2-B (face)", "face_swin_v2_base.onnx", True)):
        if (PAD_DIR / file).exists():
            out.append(CVPR24(name, label, file, face, threads))
    return out


if __name__ == "__main__":  # latency check on a synthetic frame
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    box = np.array([250, 150, 150, 180], np.float32)
    for m in light_models() + heavy_models():
        m.timed(frame, box)
        ms = [m.timed(frame, box)[1] for _ in range(10)]
        print(f"{m.label:34s} p_attack={m.predict(frame, box):.3f}  {np.median(ms):7.1f} ms")
