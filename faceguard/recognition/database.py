"""Persistent, encrypted store of enrolled people.

File layout (``faces.db``)::

    b"FGDB" | format (1 byte) | flags (1 byte, bit0 = DPAPI) | payload
    payload = [DPAPI-encrypt]( zlib( JSON ) )

Embeddings are stored column-wise as base64 float32; thumbnails as JPEG.
Writes are atomic (temp file + rename) and the previous version is kept as
``faces.db.bak``.  A damaged file is never overwritten silently: it is moved
aside and the backup is used instead.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from faceguard import paths
from faceguard.recognition.gallery import Gallery, PersonInfo
from faceguard.security import dpapi
from faceguard.vision.embedder import EMBEDDING_DIM, normalize

log = logging.getLogger(__name__)

SOURCE_CAMERA = "camera"
SOURCE_PHOTO = "photo"
SOURCE_ADAPTIVE = "adaptive"

MAX_THUMBNAILS = 6
MAX_ADAPTIVE = 30
MAGIC = b"FGDB"
FORMAT = 1
FLAG_DPAPI = 0x01


class DatabaseError(RuntimeError):
    pass


@dataclass
class Sample:
    embedding: np.ndarray
    source: str = SOURCE_CAMERA
    quality: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    created: float = field(default_factory=time.time)


@dataclass
class Person:
    id: str
    name: str
    created: float = field(default_factory=time.time)
    is_owner: bool = False
    trusted: bool = False
    samples: list[Sample] = field(default_factory=list)
    thumbnails: list[bytes] = field(default_factory=list)  # JPEG-encoded aligned faces
    last_adaptive: float = 0.0

    def info(self) -> PersonInfo:
        return PersonInfo(self.id, self.name, self.is_owner, self.trusted or self.is_owner)

    def embeddings(self, sources: Iterable[str] | None = None) -> np.ndarray:
        wanted = set(sources) if sources is not None else None
        rows = [s.embedding for s in self.samples if wanted is None or s.source in wanted]
        if not rows:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    def count(self, source: str | None = None) -> int:
        return sum(1 for s in self.samples if source is None or s.source == source)


def encode_thumbnail(aligned_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", aligned_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else b""


def decode_thumbnail(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _unb64(text: str, cols: int | None = None) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)
    return arr.reshape(-1, cols) if cols else arr


class FaceDatabase:
    def __init__(self, path: Path | None = None, encrypt: bool = True) -> None:
        self.path = Path(path) if path else paths.database_path()
        self.encrypt = bool(encrypt) and dpapi.available()
        self._lock = threading.RLock()
        self._persons: list[Person] = []
        self._version = 0
        self._snapshot: Gallery | None = None
        self._listeners: list[Callable[[], None]] = []
        self.load_warning: str | None = None

    @classmethod
    def open(cls, path: Path | None = None, encrypt: bool = True) -> "FaceDatabase":
        db = cls(path, encrypt)
        db.load()
        return db

    # ------------------------------------------------------------ persistence
    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    def load(self) -> None:
        with self._lock:
            self._persons = []
            if self.path.exists():
                try:
                    self._persons = self._read(self.path)
                except Exception as exc:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    aside = self.path.with_name(f"{self.path.name}.damaged-{stamp}")
                    os.replace(self.path, aside)
                    self.load_warning = f"Face database could not be read ({exc}); it was moved to {aside.name}."
                    log.error(self.load_warning)
                    if self.backup_path.exists():
                        try:
                            self._persons = self._read(self.backup_path)
                            self.load_warning += " Restored the previous backup."
                        except Exception as exc2:  # pragma: no cover - both damaged
                            log.error("Backup unreadable too: %s", exc2)
            self._bump()

    def _read(self, path: Path) -> list[Person]:
        raw = path.read_bytes()
        if len(raw) < 6 or raw[:4] != MAGIC:
            raise DatabaseError("not a FaceGuard database")
        if raw[4] != FORMAT:
            raise DatabaseError(f"unsupported database format {raw[4]}")
        flags, payload = raw[5], raw[6:]
        if flags & FLAG_DPAPI:
            payload = dpapi.unprotect(payload)
        doc = json.loads(zlib.decompress(payload).decode("utf-8"))
        persons = []
        for p in doc.get("persons", []):
            s = p.get("samples", {})
            embs = _unb64(s["embeddings"], EMBEDDING_DIM) if s.get("embeddings") else np.zeros((0, EMBEDDING_DIM), np.float32)
            n = len(embs)

            def col(name: str, default, _s=s, _n=n):
                values = _s.get(name) or []
                return list(values) + [default] * (_n - len(values))

            sources, quality, yaw, pitch, created = (
                col("source", SOURCE_CAMERA), col("quality", 0.0), col("yaw", 0.0), col("pitch", 0.0), col("created", 0.0)
            )
            samples = [
                Sample(normalize(embs[i]), sources[i], float(quality[i]), float(yaw[i]), float(pitch[i]), float(created[i]))
                for i in range(n)
            ]
            persons.append(
                Person(
                    id=str(p["id"]),
                    name=str(p["name"]),
                    created=float(p.get("created", 0.0)),
                    is_owner=bool(p.get("is_owner", False)),
                    trusted=bool(p.get("trusted", False)),
                    samples=samples,
                    thumbnails=[base64.b64decode(t) for t in p.get("thumbnails", [])],
                    last_adaptive=float(p.get("last_adaptive", 0.0)),
                )
            )
        return persons

    def _serialize(self) -> bytes:
        persons = []
        for p in self._persons:
            persons.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "created": p.created,
                    "is_owner": p.is_owner,
                    "trusted": p.trusted,
                    "last_adaptive": p.last_adaptive,
                    "samples": {
                        "embeddings": _b64(p.embeddings()) if p.samples else "",
                        "source": [s.source for s in p.samples],
                        "quality": [round(s.quality, 4) for s in p.samples],
                        "yaw": [round(s.yaw, 2) for s in p.samples],
                        "pitch": [round(s.pitch, 2) for s in p.samples],
                        "created": [round(s.created, 1) for s in p.samples],
                    },
                    "thumbnails": [base64.b64encode(t).decode("ascii") for t in p.thumbnails],
                }
            )
        doc = {"format": FORMAT, "app": "FaceGuard", "saved": time.time(), "persons": persons}
        payload = zlib.compress(json.dumps(doc, separators=(",", ":")).encode("utf-8"), 6)
        flags = 0
        if self.encrypt:
            payload = dpapi.protect(payload)
            flags |= FLAG_DPAPI
        return MAGIC + bytes([FORMAT, flags]) + payload

    def save(self) -> None:
        with self._lock:
            data = self._serialize()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".faces-", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                if self.path.exists():
                    os.replace(self.path, self.backup_path)
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # ------------------------------------------------------------ change tracking
    def subscribe(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _bump(self) -> None:
        self._version += 1
        self._snapshot = None

    def _commit(self) -> None:
        """Bump the version, persist, then notify listeners (outside the lock)."""
        with self._lock:
            self._bump()
            self.save()
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:  # pragma: no cover - listener bugs must not break saving
                log.exception("database listener failed")

    @property
    def version(self) -> int:
        return self._version

    # ------------------------------------------------------------ queries
    @property
    def persons(self) -> list[Person]:
        with self._lock:
            return list(self._persons)

    def persons_snapshot(self) -> list[Person]:
        """Copies that are safe to read on another thread while the database keeps changing."""
        with self._lock:
            return [replace(p, samples=list(p.samples), thumbnails=list(p.thumbnails)) for p in self._persons]

    def get(self, person_id: str) -> Person | None:
        with self._lock:
            return next((p for p in self._persons if p.id == person_id), None)

    def find(self, name: str) -> Person | None:
        key = name.strip().casefold()
        with self._lock:
            return next((p for p in self._persons if p.name.casefold() == key), None)

    def owner(self) -> Person | None:
        with self._lock:
            return next((p for p in self._persons if p.is_owner), None)

    def snapshot(self) -> Gallery:
        """Immutable gallery for matching (cached until the next change)."""
        with self._lock:
            if self._snapshot is None:
                self._snapshot = Gallery(
                    [p.info() for p in self._persons], [p.embeddings() for p in self._persons], self._version
                )
            return self._snapshot

    # ------------------------------------------------------------ mutations
    def _require(self, person_id: str) -> Person:
        p = self.get(person_id)
        if p is None:
            raise KeyError(f"No person with id {person_id}")
        return p

    def add_person(self, name: str, is_owner: bool | None = None, trusted: bool = False) -> Person:
        name = " ".join(name.split())
        if not name:
            raise ValueError("Name must not be empty")
        with self._lock:
            if self.find(name):
                raise ValueError(f"'{name}' is already enrolled")
            if is_owner is None:  # the first person enrolled owns the PC by default
                is_owner = not any(p.is_owner for p in self._persons)
            if is_owner:
                for p in self._persons:
                    p.is_owner = False
            person = Person(id=uuid.uuid4().hex, name=name, is_owner=is_owner, trusted=trusted or is_owner)
            self._persons.append(person)
        self._commit()
        return person

    def rename(self, person_id: str, name: str) -> None:
        name = " ".join(name.split())
        if not name:
            raise ValueError("Name must not be empty")
        with self._lock:
            other = self.find(name)
            if other and other.id != person_id:
                raise ValueError(f"'{name}' is already enrolled")
            self._require(person_id).name = name
        self._commit()

    def delete(self, person_id: str) -> None:
        with self._lock:
            self._persons = [p for p in self._persons if p.id != person_id]
        self._commit()

    def delete_all(self) -> None:
        with self._lock:
            self._persons = []
        self._commit()
        self.backup_path.unlink(missing_ok=True)

    def set_owner(self, person_id: str) -> None:
        with self._lock:
            target = self._require(person_id)
            for p in self._persons:
                p.is_owner = p is target
            target.trusted = True
        self._commit()

    def set_trusted(self, person_id: str, trusted: bool) -> None:
        with self._lock:
            self._require(person_id).trusted = bool(trusted)
        self._commit()

    def add_samples(self, person_id: str, samples: list[Sample], thumbnails: Iterable[np.ndarray] = ()) -> None:
        with self._lock:
            person = self._require(person_id)
            for s in samples:
                s.embedding = normalize(np.asarray(s.embedding, dtype=np.float32).reshape(EMBEDDING_DIM))
            person.samples.extend(samples)
            for t in thumbnails:
                if len(person.thumbnails) >= MAX_THUMBNAILS:
                    person.thumbnails.pop(0)
                enc = encode_thumbnail(t)
                if enc:
                    person.thumbnails.append(enc)
        self._commit()

    def add_adaptive_sample(self, person_id: str, sample: Sample, min_interval: float = 600.0) -> bool:
        """Online learning: keep a rolling window of high-confidence live samples.

        Rate-limited and capped so that the original enrollment always dominates
        the template (protects against slow template drift / poisoning).
        """
        now = time.time()
        with self._lock:
            person = self.get(person_id)
            if person is None or now - person.last_adaptive < min_interval:
                return False
            enrolled = person.count() - person.count(SOURCE_ADAPTIVE)
            cap = min(MAX_ADAPTIVE, max(5, enrolled))
            adaptive = [i for i, s in enumerate(person.samples) if s.source == SOURCE_ADAPTIVE]
            excess = len(adaptive) - cap + 1
            if excess > 0:  # drop the oldest adaptive samples (FIFO)
                drop = set(adaptive[:excess])
                person.samples = [s for i, s in enumerate(person.samples) if i not in drop]
            sample.source = SOURCE_ADAPTIVE
            sample.embedding = normalize(np.asarray(sample.embedding, dtype=np.float32).reshape(EMBEDDING_DIM))
            person.samples.append(sample)
            person.last_adaptive = now
        self._commit()
        return True

    def clear_adaptive(self, person_id: str) -> int:
        with self._lock:
            person = self._require(person_id)
            before = len(person.samples)
            person.samples = [s for s in person.samples if s.source != SOURCE_ADAPTIVE]
            removed = before - len(person.samples)
        if removed:
            self._commit()
        return removed
