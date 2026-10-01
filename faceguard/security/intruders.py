"""Intruder snapshot log: annotated JPEGs plus a JSON-lines index."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from faceguard import paths
from faceguard.security.guard import GuardFace

log = logging.getLogger(__name__)

REASON_LABELS = {
    "intruder": "Intruder at the PC",
    "shoulder": "Shoulder surfer",
    "locked": "At your locked PC",
    "manual": "Manual snapshot",
}


@dataclass
class IntruderEvent:
    time: float
    reason: str
    file: str
    faces: list[dict]

    @property
    def label(self) -> str:
        return REASON_LABELS.get(self.reason, self.reason)

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.time))


def annotate(frame: np.ndarray, boxes: list[tuple[np.ndarray, str, bool]]) -> np.ndarray:
    """Draw ``(bbox, label, authorized)`` boxes plus a timestamp on a copy of the frame."""
    img = frame.copy()
    for bbox, label, ok in boxes:
        x, y, w, h = (int(round(v)) for v in bbox[:4])
        color = (94, 197, 34) if ok else (68, 68, 239)  # BGR green / red
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, max(0, y - th - 10)), (x + tw + 10, y), color, -1)
        cv2.putText(img, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    stamp = time.strftime("FaceGuard  %Y-%m-%d %H:%M:%S")
    cv2.putText(img, stamp, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, stamp, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class IntruderLog:
    def __init__(self, folder: Path | None = None) -> None:
        self.folder = Path(folder) if folder else paths.intruders_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "events.jsonl"
        self._lock = threading.Lock()

    def record(self, frame: np.ndarray, reason: str, faces: list[GuardFace], boxes: list[tuple[np.ndarray, str, bool]]) -> IntruderEvent:
        now = time.time()
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int((now % 1) * 1000):03d}-{reason}.jpg"
        path = self.folder / name
        ok, buf = cv2.imencode(".jpg", annotate(frame, boxes), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError("could not encode snapshot")
        path.write_bytes(buf.tobytes())
        event = IntruderEvent(now, reason, name, [
            {"name": f.name, "similarity": round(f.similarity, 3), "authorized": f.authorized} for f in faces
        ])
        with self._lock, self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")
        log.info("Saved %s snapshot %s", reason, name)
        return event

    def events(self) -> list[IntruderEvent]:
        """Newest first; entries whose image was deleted are skipped."""
        out: list[IntruderEvent] = []
        if not self.index.exists():
            return out
        with self._lock:
            lines = self.index.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                d = json.loads(line)
                ev = IntruderEvent(float(d["time"]), str(d["reason"]), str(d["file"]), list(d.get("faces", [])))
            except (ValueError, KeyError, TypeError):
                continue
            if (self.folder / ev.file).exists():
                out.append(ev)
        out.sort(key=lambda e: -e.time)
        return out

    def path_of(self, event: IntruderEvent) -> Path:
        return self.folder / event.file

    def _rewrite(self, keep: list[IntruderEvent]) -> None:
        with self._lock:
            tmp = self.index.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(asdict(e)) + "\n" for e in sorted(keep, key=lambda e: e.time)), encoding="utf-8")
            tmp.replace(self.index)

    def delete(self, events: list[IntruderEvent]) -> None:
        doomed = {e.file for e in events}
        for f in doomed:
            (self.folder / f).unlink(missing_ok=True)
        self._rewrite([e for e in self.events() if e.file not in doomed])

    def clear(self) -> None:
        self.delete(self.events())

    def cleanup(self, retention_days: int) -> int:
        """Delete snapshots older than ``retention_days`` (0 = keep forever). Returns how many were removed."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        old = [e for e in self.events() if e.time < cutoff]
        if old:
            self.delete(old)
        return len(old)
