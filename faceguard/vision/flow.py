"""Planarity of tracked image features: a parallax cue that does not rely on a learned landmark detector.

The five-landmark parallax (``geometry.parallax_yaw``) is exactly zero for a planar presentation only if the
landmarks are the true projections of fixed points. A learned detector predicts where facial landmarks *should*
be, and on a foreshortened photograph it partly places them where they would be on a turned head (paper,
Section V-F). Corner features tracked by optical flow measure image motion instead: on a photograph or a screen
every tracked point lies on one plane, so the points' motion since the start of a challenge is explained by a
single homography up to tracking noise, whereas the protruding nose, receding cheeks and the face outline of a
real head leave residuals that grow with the rotation.

``PlanarityTracker`` follows up to ``max_points`` Shi-Tomasi corners inside an ellipse on the face with
pyramidal Lucas-Kanade flow, drops points that fail a forward-backward check, and reports the residual of the
best homography from the reference to the current positions, as a fraction of the inter-ocular distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Planarity:
    n: int  # points still tracked
    rms: float  # RMS residual of the least-squares homography over the RANSAC inliers / inter-ocular distance
    p75: float  # 75th percentile of the residuals of all tracked points / inter-ocular distance
    outlier_frac: float  # fraction of tracked points whose residual exceeds ``outlier_iod`` inter-ocular distances
    shift: float  # median displacement of the points since the reference frame / inter-ocular distance


class PlanarityTracker:
    def __init__(self, max_points: int = 150, min_points: int = 12, fb_max: float = 1.0,
                 outlier_iod: float = 0.05):
        self.max_points = max_points
        self.min_points = min_points
        self.fb_max = fb_max  # forward-backward error (pixels) above which a track is dropped
        self.outlier_iod = outlier_iod  # residual (fraction of the inter-ocular distance) that counts as off-plane
        self._lk = dict(winSize=(15, 15), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.reset()

    def reset(self) -> None:
        self.ref: np.ndarray | None = None
        self.cur: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.iod = 0.0

    @property
    def active(self) -> bool:
        return self.ref is not None

    def start(self, gray: np.ndarray, bbox, landmarks) -> int:
        """Pick corners on the face in ``gray`` (the reference frame); returns the number of points."""
        x, y, w, h = (float(v) for v in bbox[:4])
        lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
        self.iod = float(np.hypot(*(lm[1] - lm[0])))
        mask = np.zeros(gray.shape[:2], np.uint8)
        cx, cy = x + w / 2.0, y + h / 2.0
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(2, int(0.42 * w)), max(2, int(0.50 * h))),
                    0, 0, 360, 255, -1)
        min_dist = max(3.0, 0.03 * w)
        pts = cv2.goodFeaturesToTrack(gray, self.max_points, 0.005, min_dist, mask=mask, blockSize=5)
        if pts is None or len(pts) < self.min_points or self.iod < 1e-3:
            self.reset()
            return 0 if pts is None else len(pts)
        self.ref = pts.reshape(-1, 2).astype(np.float32)
        self.cur = self.ref.copy()
        self.prev_gray = gray
        return len(self.ref)

    def update(self, gray: np.ndarray) -> Planarity | None:
        if self.ref is None or self.prev_gray is None:
            return None
        p0 = self.cur.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self._lk)
        if p1 is None:
            self.reset()
            return None
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self._lk)
        fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_max)
        self.ref = self.ref[good]
        self.cur = p1.reshape(-1, 2)[good]
        self.prev_gray = gray
        n = len(self.ref)
        if n < self.min_points:
            return Planarity(n, math.nan, math.nan, math.nan, math.nan)
        return planarity(self.ref, self.cur, self.iod, self.outlier_iod)


def planarity(ref: np.ndarray, cur: np.ndarray, iod: float, outlier_iod: float = 0.05) -> Planarity:
    """How well one homography explains the motion ``ref -> cur`` (residuals relative to ``iod``)."""
    n = len(ref)
    shift = float(np.median(np.linalg.norm(cur - ref, axis=1))) / iod
    # A loose RANSAC (a quarter of the inter-ocular distance) only removes gross tracking failures; the fit on its
    # inliers is then the least-squares homography, whose residual is what a 3-D surface cannot avoid.
    h, inl = cv2.findHomography(ref, cur, cv2.RANSAC, 0.25 * iod, maxIters=500, confidence=0.995)
    if h is None:
        return Planarity(n, math.nan, math.nan, math.nan, shift)
    inl = inl.ravel().astype(bool)
    if inl.sum() >= 8:
        h2, _ = cv2.findHomography(ref[inl], cur[inl], 0)
        h = h2 if h2 is not None else h
    proj = cv2.perspectiveTransform(ref.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)
    res = np.linalg.norm(cur - proj, axis=1) / iod
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else math.nan
    return Planarity(n, rms, float(np.percentile(res, 75)), float(np.mean(res > outlier_iod)), shift)
