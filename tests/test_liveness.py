import math
import random

import numpy as np
import pytest

from faceguard.recognition.gallery import Gallery, Matcher, PersonInfo
from faceguard.security.liveness import (
    Challenge, LivenessConfig, LivenessObservation, LivenessSession, VerificationTask, make_sequence,
)
from tests.conftest import face_result, frame_result, identity, samples_of

FPS = 15.0


class LiveUser:
    """Simulated person who follows the on-screen instructions (reaction delay ~0.4 s)."""

    def __init__(self, rng, center, turn=26.0):
        self.rng = rng
        self.center = center
        self.turn = turn
        self.action_start = None
        self.last_instruction = None

    def react(self, state, t):
        yaw, blink, smile = 0.0, 0.05, 0.10
        if state is not None and state.phase == "challenge":
            if state.instruction != self.last_instruction:
                self.last_instruction, self.action_start = state.instruction, t
            doing = t - self.action_start
            if doing > 0.4:
                c = state.challenge
                if c is Challenge.TURN_LEFT:
                    yaw = min(self.turn, (doing - 0.4) * 60)
                elif c is Challenge.TURN_RIGHT:
                    yaw = -min(self.turn, (doing - 0.4) * 60)
                elif c is Challenge.BLINK:
                    blink = 0.85 if 0.4 < doing < 0.6 else 0.05
                elif c is Challenge.SMILE:
                    smile = 0.8
        else:
            self.last_instruction = None
        return yaw + self.rng.normal(0, 1.0), blink, smile


def run(session, behave, rng, center, embed_every=5, max_t=45.0, faces=1, swap_at=None, other=None):
    state, t, i = None, 0.0, 0
    while t < max_t:
        yaw, blink, smile = behave(state, t)
        emb = None
        if i % embed_every == 0:
            who = other if (swap_at is not None and t >= swap_at) else center
            emb = samples_of(who, 1, rng)[0]
        n = faces(t) if callable(faces) else faces
        state = session.update(LivenessObservation(t, n, yaw, blink, smile, emb, 1))
        if state.finished:
            return state
        t += 1.0 / FPS
        i += 1
    return state


@pytest.mark.parametrize("seed", range(8))
def test_sequences_always_include_a_head_turn(seed):
    rng = random.Random(seed)
    for n in (1, 2, 3, 5):
        seq = make_sequence(n, expressions=True, rng=rng)
        assert len(seq) == n and any(c.is_turn for c in seq)
        assert all(a != b for a, b in zip(seq, seq[1:]))


@pytest.mark.parametrize("seed", range(6))
def test_live_user_passes(seed):
    rng = np.random.default_rng(seed)
    center = identity(rng)
    user = LiveUser(rng, center)
    session = LivenessSession(LivenessConfig(num_challenges=3), expressions_available=True, rng=random.Random(seed))
    state = run(session, user.react, rng, center)
    assert state.phase == "passed", state.reason
    assert len(state.completed) == 3 and state.elapsed < 20


def test_printed_photo_fails():
    """A photo cannot turn its head: yaw stays ~0 whatever the attacker does with the paper."""
    rng = np.random.default_rng(1)
    center = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=2), expressions_available=False, rng=random.Random(1))
    state = run(session, lambda s, t: (rng.normal(0, 2.0), None, None), rng, center)
    assert state.phase == "failed" and ("Timed out" in state.reason or "No response" in state.reason)


def test_frozen_expression_fails_blink_challenge():
    rng = np.random.default_rng(2)
    center = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=1), expressions_available=True, rng=random.Random(0))
    session.sequence = [Challenge.BLINK]
    state = run(session, lambda s, t: (0.0, 0.05, 0.1), rng, center)
    assert state.phase == "failed"


def test_face_swap_is_detected():
    rng = np.random.default_rng(3)
    center, other = identity(rng), identity(rng)
    user = LiveUser(rng, center)
    session = LivenessSession(LivenessConfig(num_challenges=3), rng=random.Random(3))
    state = run(session, user.react, rng, center, swap_at=2.0, other=other)
    assert state.phase == "failed" and "different face" in state.reason


def test_second_person_in_view_fails():
    rng = np.random.default_rng(4)
    center = identity(rng)
    user = LiveUser(rng, center)
    session = LivenessSession(LivenessConfig(num_challenges=3), rng=random.Random(4))
    state = run(session, user.react, rng, center, faces=lambda t: 2 if t > 1.0 else 1)
    assert state.phase == "failed" and "More than one face" in state.reason


def test_face_lost_fails():
    rng = np.random.default_rng(5)
    center = identity(rng)
    user = LiveUser(rng, center)
    session = LivenessSession(LivenessConfig(num_challenges=3), rng=random.Random(5))
    state = run(session, user.react, rng, center, faces=lambda t: 0 if t > 1.5 else 1)
    assert state.phase == "failed" and "Face lost" in state.reason


def test_wrong_direction_gets_hint():
    session = LivenessSession(LivenessConfig(num_challenges=1), rng=random.Random(0))
    session.sequence = [Challenge.TURN_LEFT]
    rng = np.random.default_rng(6)
    emb = identity(rng)
    for i in range(6):
        session.update(LivenessObservation(i / FPS, 1, 0.0, None, None, emb, 1))
    assert session.phase == "challenge"
    state = session.update(LivenessObservation(0.5, 1, -30.0, None, None, None, 1))
    assert state.hint == "Other way!"


def universal_replay(t, period=1.8):
    """A recorded video of the genuine user cycling through every action: left, right, blink, smile."""
    k, phase = divmod(t, period)
    act = int(k) % 4
    doing = phase < 1.0  # 1 s action, 0.8 s back at the centre
    yaw = 26.0 if (doing and act == 0) else (-26.0 if (doing and act == 1) else 0.0)
    blink = 0.85 if (doing and act == 2 and phase < 0.2) else 0.05
    smile = 0.8 if (doing and act == 3) else 0.1
    return yaw, blink, smile


@pytest.mark.parametrize("enforce,expected", [(1, "failed"), (None, "passed")])
def test_order_enforcement_defeats_universal_replay(enforce, expected):
    rng = np.random.default_rng(10)
    center = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=3, max_wrong_actions=enforce, response_window=None),
                              expressions_available=True, rng=random.Random(0))
    session.sequence = [Challenge.BLINK, Challenge.TURN_LEFT, Challenge.SMILE]
    state = run(session, lambda s, t: universal_replay(t), rng, center)
    assert state.phase == expected, state.reason
    if expected == "failed":
        assert "Wrong action" in state.reason


def test_one_wrong_turn_is_forgiven():
    rng = np.random.default_rng(11)
    center = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=1), rng=random.Random(0))
    session.sequence = [Challenge.TURN_LEFT]

    def confused_user(state, t):
        if state is None or state.phase != "challenge":
            return 0.0, None, None
        started = confused_user.start = getattr(confused_user, "start", None) or t
        d = t - started
        return (-25.0 if 0.3 < d < 1.0 else (25.0 if d > 1.6 else 0.0)), None, None  # wrong way first, then right

    state = run(session, confused_user, rng, center)
    assert state.phase == "passed" and session.strikes == 1


@pytest.mark.parametrize("real_head,expected", [(True, "passed"), (False, "failed")])
def test_parallax_gate_rejects_a_print_that_fools_the_yaw_ratio(real_head, expected):
    from tests.conftest import FACE3D, FLAT3D, render

    rng = np.random.default_rng(12)
    emb = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=1), rng=random.Random(0))
    session.sequence = [Challenge.TURN_LEFT]
    shape = FACE3D if real_head else FLAT3D
    state, t, prompt_t = None, 0.0, None
    while t < 20 and (state is None or not state.finished):
        if state is not None and state.phase == "challenge" and prompt_t is None:
            prompt_t = t
        # reacts half a second after the prompt, and faces the camera again once the prompt is gone
        turning = state is not None and state.phase == "challenge" and prompt_t is not None and t - prompt_t > 0.5
        angle = 30.0 if turning else 0.0
        # the yaw value is forced high in both cases: only the landmarks' 3-D structure differs
        state = session.update(LivenessObservation(t, 1, angle, None, None, emb, 1, render(shape, angle)))
        t += 1 / FPS
    assert state.phase == expected


def _single_turn_session(yaw_fn, cfg=None):
    """Drive a one-challenge TURN_LEFT session; yaw_fn(seconds since the prompt appeared) -> yaw."""
    from tests.conftest import FACE3D, render

    rng = np.random.default_rng(13)
    emb = identity(rng)
    session = LivenessSession(cfg or LivenessConfig(num_challenges=1), rng=random.Random(0))
    session.sequence = [Challenge.TURN_LEFT]
    state, t, prompt_t = None, 0.0, None
    while t < 20 and (state is None or not state.finished):
        if state is not None and state.phase == "challenge" and prompt_t is None:
            prompt_t = t
        yaw = yaw_fn(t - prompt_t) if prompt_t is not None and state.phase == "challenge" else 0.0
        state = session.update(LivenessObservation(t, 1, yaw, None, None, emb, 1, render(FACE3D, yaw)))
        t += 1 / FPS
    return state, session


def test_action_already_under_way_when_the_prompt_appears_does_not_count():
    # A looping video that is turning left at the moment the prompt appears (no reaction time), then idles.
    state, _ = _single_turn_session(lambda d: 30.0 if d < 1.0 else 0.0)
    assert state.phase == "failed"


def test_no_response_within_the_window_fails():
    state, _ = _single_turn_session(lambda d: 30.0 if d > 4.5 else 0.0)
    assert state.phase == "failed" and "No response" in state.reason


def test_normal_reaction_time_passes():
    state, _ = _single_turn_session(lambda d: min(30.0, max(0.0, (d - 0.7) * 90)))
    assert state.phase == "passed"


def test_hesitant_wrong_turn_counts_as_one_strike():
    # wrong way, hovering around the threshold (in and out several times), then the right way
    def yaw(d):
        if d < 0.4:
            return 0.0
        if d < 2.0:
            return -18.0 + 3.0 * math.sin(d * 25)
        return min(30.0, (d - 2.3) * 90) if d > 2.3 else 0.0

    state, session = _single_turn_session(yaw)
    assert state.phase == "passed" and session.strikes == 1


def _verify(seed, enrolled_center, probe_center, require_owner=False, owner=True):
    rng = np.random.default_rng(seed)
    gallery = Gallery([PersonInfo("a", "Alice", is_owner=owner)], [samples_of(enrolled_center, 12, rng)])
    task = VerificationTask(gallery, Matcher(0.45, 0.30), LivenessConfig(num_challenges=2), False, require_owner,
                            random.Random(seed))
    user = LiveUser(rng, probe_center)
    state, t, i = None, 0.0, 0
    while t < 40 and task.result is None:
        yaw, _, _ = user.react(state, t)
        fresh = i % 5 == 0
        emb = samples_of(probe_center, 1, rng)[0]
        state = task.update(frame_result(t, [face_result(yaw=yaw, embedding=emb, fresh=fresh)]))
        t += 1 / FPS
        i += 1
    return task.result


def test_verification_of_enrolled_live_user():
    rng = np.random.default_rng(7)
    c = identity(rng)
    r = _verify(7, c, c)
    assert r.live and r.verified and r.match.person.name == "Alice"


def test_live_stranger_is_not_verified():
    rng = np.random.default_rng(8)
    r = _verify(8, identity(rng), identity(rng))
    assert r.live and not r.verified and r.reason == "Face not recognised"


def test_require_owner():
    rng = np.random.default_rng(9)
    c = identity(rng)
    r = _verify(9, c, c, require_owner=True, owner=False)
    assert r.live and not r.verified and "not the owner" in r.reason


def test_identity_check_skipped_on_strongly_turned_faces():
    """Real-world pilot: genuine users failed 'A different face appeared' at 31-49 deg, where embeddings drift."""
    rng = np.random.default_rng(14)
    center, other = identity(rng), identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=1, response_window=None), rng=random.Random(0))
    session.sequence = [Challenge.TURN_LEFT]
    t = 0.0
    for _ in range(8):  # align with a frontal reference
        session.update(LivenessObservation(t, 1, 0.0, None, None, samples_of(center, 1, rng)[0], 1))
        t += 1 / FPS
    # a dissimilar embedding at 40 deg (as a real turned face can produce) must not fail the session
    state = session.update(LivenessObservation(t, 1, 40.0, None, None, samples_of(other, 1, rng)[0], 1))
    assert state.phase != "failed"
    # ... but the same embedding near the baseline pose still does
    state = session.update(LivenessObservation(t + 0.1, 1, 5.0, None, None, samples_of(other, 1, rng)[0], 1))
    assert state.phase == "failed" and "different face" in state.reason


def test_natural_blink_peak_of_a_webcam_is_accepted():
    """At webcam frame rates a natural blink is sampled near 0.3-0.4, below the former 0.45 threshold."""
    rng = np.random.default_rng(15)
    emb = identity(rng)
    session = LivenessSession(LivenessConfig(num_challenges=1), expressions_available=True, rng=random.Random(0))
    session.sequence = [Challenge.BLINK]
    state, t, prompt_t = None, 0.0, None
    while t < 10 and (state is None or not state.finished):
        if state is not None and state.phase == "challenge" and prompt_t is None:
            prompt_t = t
        d = None if prompt_t is None else t - prompt_t
        blink = 0.36 if (d is not None and 0.6 < d < 0.75) else 0.04
        state = session.update(LivenessObservation(t, 1, 0.0, blink, 0.05, emb, 1))
        t += 0.1
    assert state.phase == "passed", state.reason
