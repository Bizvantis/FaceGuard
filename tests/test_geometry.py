import math

import numpy as np
import pytest

from faceguard.vision.geometry import iou, parallax_yaw, pose_from_landmarks
from tests.conftest import FACE3D, FLAT3D, render

# right eye, left eye, nose, right mouth corner, left mouth corner (raw camera image, subject facing camera)
FRONTAL = np.array([[40, 50], [80, 50], [60, 70.8], [45, 90], [75, 90]], dtype=float)


def rotate(points, degrees, center=(60, 70)):
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (points - center) @ r.T + center


def test_frontal_face_has_near_zero_pose():
    p = pose_from_landmarks(FRONTAL)
    assert abs(p.yaw) < 0.5 and abs(p.pitch) < 0.5 and abs(p.roll) < 0.5


def test_nose_towards_image_right_means_subject_turned_left():
    lm = FRONTAL.copy()
    lm[2, 0] += 12  # nose moves right in the raw image
    assert pose_from_landmarks(lm).yaw > 10


def test_in_plane_rotation_does_not_change_yaw():
    lm = FRONTAL.copy()
    lm[2, 0] += 8
    base = pose_from_landmarks(lm)
    rolled = pose_from_landmarks(rotate(lm, 25))
    assert rolled.yaw == pytest.approx(base.yaw, abs=1e-6)
    assert rolled.roll == pytest.approx(25, abs=1e-6)


def test_mirrored_image_flips_yaw():
    lm = FRONTAL.copy()
    lm[2, 0] += 10
    mirrored = lm.copy()
    mirrored[:, 0] = 120 - mirrored[:, 0]
    mirrored = mirrored[[1, 0, 2, 4, 3]]  # the detector still lists the image-left eye first
    assert pose_from_landmarks(mirrored).yaw == pytest.approx(-pose_from_landmarks(lm).yaw, abs=1e-6)


def test_flat_photo_squeezed_horizontally_keeps_yaw():
    """A photo rotated about its vertical axis is an affine squeeze: yaw must stay ~0 (liveness relies on it)."""
    squeezed = FRONTAL.copy()
    squeezed[:, 0] = 60 + (squeezed[:, 0] - 60) * math.cos(math.radians(50))
    assert abs(pose_from_landmarks(squeezed).yaw) < 0.5


def test_pitch_positive_when_looking_up():
    lm = FRONTAL.copy()
    lm[2, 1] -= 6  # nose moves towards the eye line
    assert pose_from_landmarks(lm).pitch > 10


@pytest.mark.parametrize("tilt", [-60, -30, 15, 45, 60])
def test_parallax_is_zero_for_a_tilted_print(tilt):
    """However a flat photo is tilted (strong perspective included), its nose follows the plane exactly."""
    assert abs(parallax_yaw(render(FLAT3D), render(FLAT3D, tilt))) < 1e-3


def test_parallax_is_zero_for_any_planar_motion():
    rng = np.random.default_rng(0)
    base = render(FLAT3D)
    for _ in range(20):
        h = np.eye(3) + rng.normal(0, [[0.1, 0.1, 20], [0.1, 0.1, 20], [3e-4, 3e-4, 0]])
        p = np.c_[base, np.ones(5)] @ h.T
        assert abs(parallax_yaw(base, p[:, :2] / p[:, 2:])) < 1e-2


@pytest.mark.parametrize("turn", [-30, -20, 20, 30])
def test_parallax_tracks_a_real_head_turn(turn):
    est = parallax_yaw(render(FACE3D), render(FACE3D, turn))
    assert math.copysign(1, est) == math.copysign(1, turn)
    assert abs(est - turn) < 0.35 * abs(turn)


def test_iou():
    a = np.array([0, 0, 10, 10])
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, np.array([20, 20, 5, 5])) == 0.0
    assert iou(a, np.array([5, 0, 10, 10])) == pytest.approx(50 / 150)
