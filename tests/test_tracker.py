import numpy as np

from faceguard.recognition.tracker import FaceTracker
from faceguard.vision.detector import Face


def face(x, y, w=100):
    return Face(np.array([x, y, w, w], np.float32), np.zeros((5, 2), np.float32), 0.9)


def test_track_id_is_stable_while_face_moves():
    tr = FaceTracker()
    ids = [tr.update([face(100 + 5 * i, 100)], i * 0.03)[0].id for i in range(20)]
    assert len(set(ids)) == 1


def test_fast_motion_with_little_overlap_keeps_identity():
    tr = FaceTracker()
    a = tr.update([face(100, 100)], 0.0)[0].id
    b = tr.update([face(140, 140)], 0.25)[0].id  # IoU 0.22 (< 0.25) but centres only 57 px apart
    assert a == b


def test_two_faces_two_tracks_and_expiry():
    tr = FaceTracker(max_missed=2)
    out = tr.update([face(0, 0), face(400, 0)], 0.0)
    assert len({t.id for t in out}) == 2
    for i in range(1, 5):
        tr.update([face(0, 0)], i * 0.05)
    assert len(tr.tracks) == 1


def test_new_person_gets_new_id():
    tr = FaceTracker()
    first = tr.update([face(0, 0)], 0.0)[0].id
    for i in range(1, 8):
        tr.update([], i * 0.1)
    assert tr.update([face(0, 0)], 1.0)[0].id != first
