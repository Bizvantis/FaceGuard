"""Online (adaptive) learning of face templates.

Appearance drifts - haircuts, glasses, beards, seasons, a new desk lamp.  When
a person is recognised with a *very* high score in a good-quality frame, the
live embedding is occasionally added to their template (at most once every 10
minutes, capped, never outnumbering the original enrollment - see
``FaceDatabase.add_adaptive_sample``).  Only embeddings that add new
information (not near-duplicates of existing samples) are kept.
"""

from __future__ import annotations

import logging

import numpy as np

from faceguard.recognition.database import FaceDatabase, Sample
from faceguard.recognition.engine import FrameResult
from faceguard.recognition.gallery import Gallery

log = logging.getLogger(__name__)


class AdaptiveLearner:
    def __init__(self, db: FaceDatabase, margin: float = 0.12, novelty: float = 0.92, min_interval: float = 600.0):
        self.db = db
        self.margin = margin
        self.novelty = novelty
        self.min_interval = min_interval

    def observe(self, result: FrameResult, gallery: Gallery, threshold: float) -> str | None:
        """Returns the person's name if a sample was added."""
        for f in result.faces:
            m = f.match
            if not (f.fresh and m.is_match and m.person and f.embedding is not None):
                continue
            q = f.quality
            if q is None or not q.ok or f.embed_count < 3 or abs(q.yaw) > 25 or q.face_px < 80:
                continue
            if m.similarity < threshold + self.margin:
                continue
            j = gallery.index_of(m.person.id)
            if j is None:
                continue
            s, e = gallery.slices[j]
            if float(np.max(gallery.matrix[s:e] @ f.embedding)) >= self.novelty:
                continue  # nothing new to learn from this frame
            sample = Sample(f.embedding.copy(), quality=q.score, yaw=q.yaw, pitch=q.pitch)
            if self.db.add_adaptive_sample(m.person.id, sample, self.min_interval):
                log.info("Adaptive learning: added a sample for %s (score %.2f)", m.person.name, m.similarity)
                return m.person.name
        return None
