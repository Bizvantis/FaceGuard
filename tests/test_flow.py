import math

import cv2
import numpy as np

from faceguard.security.liveness import Challenge, LivenessConfig, LivenessObservation, LivenessSession
from faceguard.vision.flow import PlanarityTracker, planarity

K = np.array([[457.0, 0, 320], [0, 457.0, 240], [0, 0, 1]])


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (60, 80), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def _rot_y(deg):
    b = math.radians(deg)
    return np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])


def _plane_homography(deg, d=500.0):
    c, n = np.array([0, 0, d]), np.array([0, 0, 1.0])
    r = _rot_y(deg)
    return K @ (r + np.outer((np.eye(3) - r) @ c, n) / d) @ np.linalg.inv(K)


def test_tilted_plane_is_planar():
    img = _texture()
    tr = PlanarityTracker()
    lm = np.array([[290, 210], [350, 210], [320, 245], [298, 280], [342, 280]], float)
    assert tr.start(img, [240, 150, 160, 180], lm) >= 12
    p = None
    for k in range(1, 21):
        p = tr.update(cv2.warpPerspective(img, _plane_homography(1.5 * k), (640, 480)))
    assert p is not None and p.n >= 12
    assert p.rms < 0.03  # tracking noise only: about 1 % of the inter-ocular distance


def test_points_off_a_plane_are_detected():
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-70, 70, 120), rng.uniform(-90, 90, 120)]
    depth = 40.0 * np.exp(-(pts[:, 0] ** 2 + pts[:, 1] ** 2) / (2 * 40.0 ** 2))  # a nose-like bump (mm)
    world = np.c_[pts, 500.0 - depth]

    def project(r):
        x = (world - [0, 0, 500.0]) @ r.T + [0, 0, 500.0]
        uv = x @ K.T
        return uv[:, :2] / uv[:, 2:]

    ref = project(np.eye(3)).astype(np.float32)
    flat = planarity(ref, project(_rot_y(0)).astype(np.float32) + 0.0, 60.0)
    turned = planarity(ref, project(_rot_y(20)).astype(np.float32), 60.0)
    assert turned.rms > 5 * max(flat.rms, 1e-3)


def test_flow_gate_blocks_planar_turns():
    cfg = LivenessConfig(use_expressions=False, num_challenges=1, parallax_fraction=None, flow_threshold=0.05,
                         response_window=None)
    s = LivenessSession(cfg, False)
    s.sequence = [Challenge.TURN_LEFT]
    t = 0.0
    for _ in range(6):
        s.update(LivenessObservation(t, 1, yaw=0.0, embedding=np.ones(128)))
        t += 0.05
    for _ in range(40):  # the yaw says "turned", but the tracked points stay on one plane
        st = s.update(LivenessObservation(t, 1, yaw=30.0, planarity=0.01))
        t += 0.05
    assert Challenge.TURN_LEFT not in st.completed
    for _ in range(4):
        st = s.update(LivenessObservation(t, 1, yaw=30.0, planarity=0.08))
        t += 0.05
    assert Challenge.TURN_LEFT in st.completed
