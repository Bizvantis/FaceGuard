import pytest

from faceguard.recognition.gallery import MatchResult, PersonInfo
from faceguard.security.guard import GuardConfig, GuardFace, GuardState, PresenceGuard, guard_faces
from tests.conftest import face_result, frame_result

OWNER = GuardFace(1, authorized=True, stranger=False, name="Me", similarity=0.8, face_px=150, quality_ok=True)
STRANGER = GuardFace(2, authorized=False, stranger=True, name=None, similarity=0.1, face_px=150, quality_ok=True)
FAR_STRANGER = GuardFace(3, authorized=False, stranger=True, name=None, similarity=0.1, face_px=45, quality_ok=True)
DT = 0.25


def simulate(guard, until, faces_at, idle_at=lambda t: 999.0, start=0.0, owner=True):
    """Run the guard from ``start`` to ``until``; returns [(t, action)]. Lock actions lock the 'session'."""
    out, t = [], start
    while t <= until + 1e-9:
        for a in guard.update(t, faces_at(t), idle_at(t), owner):
            out.append((round(t, 2), a))
            if a.kind == "lock":
                guard.on_session_locked(t)  # what Windows does after LockWorkStation()
        t += DT
    return out


def kinds(actions):
    return [a.kind for _, a in actions]


def started(**cfg):
    g = PresenceGuard(GuardConfig(**cfg))
    g.start(0.0)
    return g


def test_walk_away_warns_then_locks():
    g = started(absence_timeout=30, warn_before_lock=8)
    acts = simulate(g, 40, lambda t: [])
    assert kinds(acts) == ["warn", "lock"]
    warn_t, lock_t = acts[0][0], acts[1][0]
    assert 21.9 <= warn_t <= 22.5 and 29.9 <= lock_t <= 30.3


def test_keyboard_activity_prevents_absence_lock():
    g = started(absence_timeout=30)
    assert simulate(g, 120, lambda t: [], idle_at=lambda t: 2.0) == []


def test_owner_presence_resets_countdown():
    g = started(absence_timeout=30)
    acts = simulate(g, 60, lambda t: [OWNER] if t < 20 else [])
    lock_times = [t for t, a in acts if a.kind == "lock"]
    assert lock_times == [49.75]  # owner last seen at 19.75 s + 30 s timeout
    assert g.state == GuardState.LOCKED


def test_intruder_is_captured_and_locked_after_confirmation():
    g = started(intruder_seconds=2.5)
    acts = simulate(g, 10, lambda t: [STRANGER] if t >= 5 else [OWNER])
    assert kinds(acts) == ["snapshot", "notify", "lock"]
    first = acts[0][0]
    assert 5 + 2.5 <= first <= 5 + 3.0 + DT  # confirmed after 2.5 s, owner gone for 3 s
    assert acts[1][1].title == "Intruder detected"
    assert g.state == GuardState.LOCKED


def test_far_away_or_blurry_faces_never_lock():
    g = started(absence_timeout=1000)
    blurry = GuardFace(4, False, True, None, 0.1, 150, quality_ok=False)
    assert simulate(g, 30, lambda t: [FAR_STRANGER, blurry], idle_at=lambda t: 0.0) == []


def test_shoulder_surfer_alert_without_lock():
    g = started(shoulder_seconds=3.0)
    acts = simulate(g, 10, lambda t: [OWNER, FAR_STRANGER])
    assert kinds(acts) == ["notify", "snapshot"]
    assert acts[0][1].reason == "shoulder"
    assert g.state == GuardState.PRESENT


def test_grace_after_unlock_suggests_reenrolling_instead_of_locking():
    g = started()
    g.on_session_locked(10.0)
    assert simulate(g, 20, lambda t: [STRANGER], start=10.0) == []  # locked: no actions
    g.on_session_unlocked(20.0)
    acts = simulate(g, 30, lambda t: [STRANGER], start=20.0)
    assert kinds(acts) == ["notify"] and acts[0][1].reason == "unrecognized"


def test_intruder_locks_are_rate_limited():
    """Someone who keeps unlocking (i.e. knows the password) must not be locked out in a loop."""
    g = started(alert_cooldown=60, max_intruder_locks=3, intruder_window=600, grace_after_unlock=15)
    actions, t, locked_at = [], 0.0, None
    while t <= 500:
        if locked_at is not None and t >= locked_at + 20:
            g.on_session_unlocked(t)
            locked_at = None
        for a in g.update(t, [STRANGER], 999.0, True):
            actions.append(a)
            if a.kind == "lock":
                g.on_session_locked(t)
                locked_at = t
        t += DT
    assert sum(a.kind == "lock" for a in actions) == 3
    assert any(a.reason == "rate_limited" for a in actions)
    assert any(a.reason == "unrecognized" for a in actions)


def test_no_owner_enrolled_does_nothing():
    g = started()
    assert simulate(g, 100, lambda t: [STRANGER], owner=False) == []
    assert g.state == GuardState.NO_OWNER


def test_camera_outage_never_causes_instant_lock():
    g = started(absence_timeout=30)
    simulate(g, 5, lambda t: [OWNER])
    g.on_camera_lost(5.0)
    assert simulate(g, 100, lambda t: [], start=5.0) == []
    g.on_camera_restored(100.0)
    acts = simulate(g, 110, lambda t: [], start=100.0)
    assert "lock" not in kinds(acts)


def test_locked_pc_watch_photographs_strangers_without_locking_again():
    g = started(watch_locked_minutes=30, intruder_seconds=2.5, alert_cooldown=60)
    g.on_session_locked(10.0)
    assert g.watching_locked(10.0) and g.status_text(10.0) == "PC locked - watching for intruders"
    acts, t = [], 10.0
    while t <= 20.0:
        acts += [(t, a) for a in g.update_locked(t, [STRANGER])]
        t += DT
    assert [a.kind for _, a in acts] == ["snapshot", "notify"]  # once per cooldown, never a lock
    assert acts[0][0] >= 12.5  # only after the stranger was seen for intruder_seconds
    assert acts[0][1].reason == "locked" and acts[1][1].title == "Someone was at your locked PC"
    assert g.state == GuardState.LOCKED


def test_locked_watch_ignores_the_owner_and_expires():
    g = started(watch_locked_minutes=1)
    g.on_session_locked(0.0)
    assert all(g.update_locked(t / 4, [OWNER]) == [] for t in range(0, 200))  # owner coming back: nothing
    assert not g.watching_locked(61.0)
    assert all(g.update_locked(61 + t / 4, [STRANGER]) == [] for t in range(40))  # window closed
    assert g.status_text(61.0) == "PC locked"


def test_locked_watch_can_be_switched_off():
    g = started(watch_locked_minutes=0)
    g.on_session_locked(0.0)
    assert not g.watching_locked(1.0)
    assert all(g.update_locked(t / 4, [STRANGER]) == [] for t in range(40))


def test_pause_survives_locking_and_unlocking():
    g = started()
    g.pause(0.0, seconds=900)
    g.on_session_locked(10.0)
    assert g.state == GuardState.PAUSED
    g.on_session_unlocked(20.0)
    assert g.state == GuardState.PAUSED


def test_pause_with_timeout_resumes():
    g = started(absence_timeout=10)
    g.pause(0.0, seconds=60)
    assert simulate(g, 59, lambda t: []) == []
    acts = simulate(g, 80, lambda t: [], start=60.0)
    assert "lock" in kinds(acts)


@pytest.mark.parametrize("status,authorized,trusted,expect_stranger", [
    ("match", True, True, False),
    ("match", False, False, True),  # enrolled but not trusted -> treated as a stranger
    ("unknown", False, False, True),
    ("uncertain", False, False, False),
])
def test_guard_faces_translation(status, authorized, trusted, expect_stranger):
    person = PersonInfo("p", "Pat", is_owner=authorized, trusted=trusted)
    f = face_result(match=MatchResult(status, person, 0.5))
    gf = guard_faces(frame_result(0.0, [f]))[0]
    assert gf.stranger == expect_stranger
    assert gf.authorized == (status == "match" and authorized)


def test_single_embedding_is_not_enough_to_call_someone_a_stranger():
    f = face_result(match=MatchResult("unknown", None, 0.1), embed_count=1)
    assert not guard_faces(frame_result(0.0, [f]))[0].stranger
