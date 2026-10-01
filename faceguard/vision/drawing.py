"""OpenCV overlays for the command-line windows (the GUI draws its own with Qt)."""

from __future__ import annotations

import cv2
import numpy as np

from faceguard.recognition.engine import FaceResult, FrameResult

GREEN = (94, 197, 34)
AMBER = (11, 158, 245)
RED = (68, 68, 239)
BLUE = (255, 140, 79)
WHITE = (255, 255, 255)
DARK = (23, 17, 15)


def color_for(face: FaceResult) -> tuple[int, int, int]:
    m = face.match
    if m.is_match:
        return GREEN if (m.person and m.person.authorized) else AMBER
    if m.is_unknown:
        return RED
    return AMBER if m.status == "uncertain" else BLUE


def _mx(x: float, width: int, mirror: bool) -> float:
    return width - x if mirror else x


def label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.6) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + base - 2), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def draw_results(frame: np.ndarray, result: FrameResult, mirror: bool = True, landmarks: bool = True) -> np.ndarray:
    img = cv2.flip(frame, 1) if mirror else frame.copy()
    w = img.shape[1]
    for f in result.faces:
        x, y, bw, bh = (float(v) for v in f.bbox)
        x0 = _mx(x + bw, w, mirror) if mirror else x
        c = color_for(f)
        p0, p1 = (int(x0), int(y)), (int(x0 + bw), int(y + bh))
        cv2.rectangle(img, p0, p1, c, 2, cv2.LINE_AA)
        text = f.match.label
        if f.match.status in ("match", "uncertain", "unknown") and f.match.person is not None:
            text += f"  {f.match.similarity:.2f}"
        label(img, text, (p0[0], p0[1] - 4), c)
        if landmarks:
            for px, py in f.landmarks:
                cv2.circle(img, (int(_mx(px, w, mirror)), int(py)), 2, WHITE, -1, cv2.LINE_AA)
    return img


def panel(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 12), scale: float = 0.6) -> None:
    if not lines:
        return
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 10
    width = max(s[0] for s in sizes) + 20
    x, y = origin
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + lh * len(lines) + 8), DARK, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + 10, y + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def progress_bar(img: np.ndarray, fraction: float, color=GREEN, height: int = 10) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - height), (w, h), DARK, -1)
    cv2.rectangle(img, (0, h - height), (int(w * max(0.0, min(1.0, fraction))), h), color, -1)
