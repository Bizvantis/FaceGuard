import numpy as np

from faceguard.recognition.enrollment import DEFAULT_TARGETS, EnrollmentSession, find_duplicate
from faceguard.recognition.gallery import Gallery, Matcher, PersonInfo
from tests.conftest import face_result, frame_result, identity, ok_quality, samples_of

FPS = 15.0
FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeEngine:
    """Stands in for FaceEngine.embed_face: returns embeddings of one simulated person."""

    def __init__(self, center, rng, issues=()):
        self.center, self.rng, self.issues = center, rng, issues

    def embed_face(self, frame, face, tta=True, thresholds=None):
        return np.zeros((112, 112, 3), np.uint8), samples_of(self.center, 1, self.rng)[0], ok_quality(issues=self.issues)


def pose_for(key, base_pitch=3.0):
    return {
        "center": (0.0, base_pitch), "left": (22.0, base_pitch), "right": (-22.0, base_pitch),
        "up": (0.0, base_pitch + 12.0), "down": (0.0, base_pitch - 12.0),
    }[key]


def run(session, engine, faces_fn, max_t=90.0):
    t, state = 0.0, None
    while t < max_t:
        key = session.current.key if session.current else "center"
        state = session.update(FRAME, frame_result(t, faces_fn(key, t)), engine, t)
        if state.phase != "running":
            break
        t += 1.0 / FPS
    return state, t


def test_cooperative_user_completes_all_poses():
    rng = np.random.default_rng(0)
    center = identity(rng)
    session = EnrollmentSession("Alice")
    state, t = run(session, FakeEngine(center, rng), lambda key, t: [face_result(yaw=pose_for(key)[0], pitch=pose_for(key)[1])])
    assert state.phase == "done" and t < 15
    result = session.finish()
    assert len(result.samples) == sum(tg.count for tg in DEFAULT_TARGETS) == 22
    assert len(result.thumbnails) == 3 and result.skipped == []
    assert result.consistency > 0.7
    yaws = [s.yaw for s in result.samples]
    assert max(yaws) > 15 and min(yaws) < -15  # pose diversity really was captured


def test_pose_the_user_cannot_do_is_skipped_after_timeout():
    rng = np.random.default_rng(1)
    session = EnrollmentSession("Bob", target_timeout=5.0)

    def never_left(key, t):
        yaw, pitch = pose_for(key)
        return [face_result(yaw=0.0 if key == "left" else yaw, pitch=pitch)]

    state, _ = run(session, FakeEngine(identity(rng), rng), never_left)
    assert state.phase == "done"
    assert session.finish().skipped == ["Left"]


def test_two_people_in_view_blocks_capture():
    rng = np.random.default_rng(2)
    session = EnrollmentSession("Carol")
    engine = FakeEngine(identity(rng), rng)
    state = session.update(FRAME, frame_result(0.0, [face_result(x=100), face_result(track_id=2, x=600)]), engine, 0.0)
    assert not state.captured and "one person" in state.hint


def test_quality_problem_is_reported_and_blocks_capture():
    rng = np.random.default_rng(3)
    session = EnrollmentSession("Dan")
    engine = FakeEngine(identity(rng), rng, issues=("Too dark - add some light in front of you",))
    state = session.update(FRAME, frame_result(0.0, [face_result()]), engine, 0.0)
    assert not state.captured and state.hint.startswith("Too dark")


def test_duplicate_detection():
    rng = np.random.default_rng(4)
    alice = identity(rng)
    gallery = Gallery([PersonInfo("a", "Alice")], [samples_of(alice, 10, rng)])
    matcher = Matcher(0.45, 0.30)
    dup = find_duplicate(gallery, matcher, samples_of(alice, 1, rng)[0])
    assert dup is not None and dup.person.name == "Alice"
    assert find_duplicate(gallery, matcher, samples_of(alice, 1, rng)[0], exclude_id="a") is None
    assert find_duplicate(gallery, matcher, identity(rng)) is None
