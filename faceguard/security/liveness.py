"""Active liveness detection (anti-spoofing) by randomised challenge-response.

Why this works against photos and screens
-----------------------------------------
* **Head turns** are measured from the nose position relative to the eyes.
  Tilting or rotating a flat photo compresses the whole face uniformly, so the
  nose stays centred between the eyes; only a real 3-D head produces the
  required ~18 deg yaw change (see ``vision/geometry.py``).
* **Blink / smile** use MediaPipe blendshape scores - a still image cannot
  change expression.
* The sequence is random and time-limited, so a pre-recorded video would have
  to contain exactly the right actions in the right order.
* The same face must stay in view the whole time (embedding consistency
  check), so an attacker cannot swap a photo for a live face mid-way.

Limitation (documented in the README): a skilled attacker with a realistic
3-D mask or a live deepfake could still pass - only depth/IR cameras such as
those required by Windows Hello can defeat those.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from faceguard.recognition.engine import FrameResult
from faceguard.recognition.gallery import Gallery, Matcher, MatchResult
from faceguard.vision.embedder import normalize
from faceguard.vision.geometry import parallax_yaw


class Challenge(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BLINK = "blink"
    SMILE = "smile"

    @property
    def instruction(self) -> str:
        return {
            Challenge.TURN_LEFT: "Turn your head to the LEFT",
            Challenge.TURN_RIGHT: "Turn your head to the RIGHT",
            Challenge.BLINK: "BLINK your eyes",
            Challenge.SMILE: "SMILE",
        }[self]

    @property
    def is_turn(self) -> bool:
        return self in (Challenge.TURN_LEFT, Challenge.TURN_RIGHT)


@dataclass
class LivenessConfig:
    num_challenges: int = 3
    challenge_timeout: float = 7.0
    turn_degrees: float = 18.0
    center_degrees: float = 12.0
    hold_frames: int = 2
    use_expressions: bool = True
    max_face_loss: float = 1.0
    min_same_person: float = 0.40
    continuity_max_yaw: float = 20.0  # identity continuity is checked only on embeddings this close to the baseline pose
    blink_closed: float = 0.30  # blink accepted when the score rises above max(blink_closed, base + blink_closed_rel)
    blink_closed_rel: float = 0.20
    blink_open: float = 0.20  # ... and falls again below max(blink_open, base + blink_open_rel)
    blink_open_rel: float = 0.10
    baseline_frames: int = 4
    total_timeout: float = 45.0
    max_wrong_actions: int | None = 1  # None disables order enforcement (used for the ablation study)
    penalize_smiles: bool = True  # a smile during a turn/blink prompt is also a wrong action (blinks never are)
    min_response: float = 0.15  # an action that starts sooner after the prompt cannot be a reaction to it
    response_window: float | None = 3.5  # the requested action must start within this many seconds (None = off)
    parallax_fraction: float | None = 0.8  # turns also need nose parallax >= this x turn_degrees (None = off)


@dataclass
class LivenessObservation:
    t: float
    num_faces: int
    yaw: float = 0.0
    blink: float | None = None
    smile: float | None = None
    embedding: np.ndarray | None = None  # only on frames where a fresh embedding was computed
    track_id: int | None = None
    landmarks: np.ndarray | None = None  # the 5 facial landmarks (enables the planar-attack parallax test)


@dataclass
class LivenessState:
    phase: str  # align | challenge | recenter | final | passed | failed
    instruction: str
    hint: str
    step: int
    total: int
    challenge: Challenge | None
    time_left: float
    progress: float
    meter: float  # 0..1 progress towards completing the current action
    completed: list[Challenge] = field(default_factory=list)
    reason: str = ""
    elapsed: float = 0.0

    @property
    def finished(self) -> bool:
        return self.phase in ("passed", "failed")


def make_sequence(n: int, expressions: bool, rng: random.Random) -> list[Challenge]:
    """Random challenge order with at least one head turn and no immediate repeats."""
    pool = [Challenge.TURN_LEFT, Challenge.TURN_RIGHT] + ([Challenge.BLINK, Challenge.SMILE] if expressions else [])
    n = max(1, n)
    while True:
        seq: list[Challenge] = []
        bag: list[Challenge] = []
        while len(seq) < n:
            if not bag:
                bag = pool[:]
                rng.shuffle(bag)
            c = bag.pop()
            if seq and seq[-1] == c:
                continue
            seq.append(c)
        if any(c.is_turn for c in seq):
            return seq


class LivenessSession:
    def __init__(self, config: LivenessConfig | None = None, expressions_available: bool = False,
                 rng: random.Random | None = None, t0: float | None = None) -> None:
        self.cfg = config or LivenessConfig()
        self.expressions = expressions_available and self.cfg.use_expressions
        self.sequence = make_sequence(self.cfg.num_challenges, self.expressions, rng or random.SystemRandom())
        self.phase = "align"
        self.reason = ""
        self.completed: list[Challenge] = []
        self._i = 0
        self._t0 = t0
        self._phase_t = t0
        self._last_face_t = t0
        self._multi_since: float | None = None
        self._streak = 0
        self._wrong_streak = 0
        self._in_wrong = False
        self._prev_progress = 0.0
        self._onset_ok = False
        self._responded = False
        self.strikes = 0
        self._baseline: list[tuple[float, float | None, float | None]] = []
        self._baseline_landmarks: list[np.ndarray] = []
        self.base_landmarks: np.ndarray | None = None
        self.base_yaw = 0.0
        self.base_blink = 0.0
        self.base_smile = 0.0
        self.base_embedding: np.ndarray | None = None
        self.embeddings: list[np.ndarray] = []  # frontal fresh embeddings (for the final identity decision)
        self._blink_closed = False
        self._track_id: int | None = None
        self._need_reverify = False
        self._hint = ""
        self._meter = 0.0

    # ------------------------------------------------------------ helpers
    @property
    def challenge(self) -> Challenge | None:
        return self.sequence[self._i] if self._i < len(self.sequence) and self.phase in ("challenge", "recenter") else None

    def _fail(self, reason: str) -> None:
        self.phase = "failed"
        self.reason = reason

    def _set_phase(self, phase: str, t: float) -> None:
        self.phase = phase
        self._phase_t = t
        self._streak = 0
        self._wrong_streak = 0
        self._in_wrong = False
        self._prev_progress = 0.0
        self._onset_ok = False
        self._responded = False
        self._blink_closed = False

    def state(self, t: float) -> LivenessState:
        if self.phase == "align":
            instruction = "Look straight at the camera"
        elif self.phase == "challenge":
            instruction = self.sequence[self._i].instruction
        elif self.phase in ("recenter", "final"):
            instruction = "Look back at the camera"
        elif self.phase == "passed":
            instruction = "Liveness confirmed"
        else:
            instruction = "Liveness check failed"
        limit = self.cfg.challenge_timeout
        time_left = max(0.0, limit - (t - self._phase_t)) if self.phase in ("challenge", "recenter", "final") else 0.0
        done = len(self.completed) + (1 if self.phase == "passed" else 0)
        progress = min(1.0, done / (len(self.sequence) + 1))
        elapsed = t - self._t0 if self._t0 is not None else 0.0
        return LivenessState(
            self.phase, instruction, self._hint, min(self._i + 1, len(self.sequence)), len(self.sequence),
            self.challenge, time_left, progress, self._meter, list(self.completed), self.reason, elapsed,
        )

    def _centered(self, obs: LivenessObservation) -> bool:
        if abs(obs.yaw - self.base_yaw) > self.cfg.center_degrees:
            return False
        if obs.smile is not None and obs.smile > self.base_smile + 0.25:
            return False
        return True

    # ------------------------------------------------------------ main update
    def update(self, obs: LivenessObservation) -> LivenessState:
        t = obs.t
        if self._t0 is None:
            self._t0 = self._phase_t = self._last_face_t = t
        if self.phase in ("passed", "failed"):
            return self.state(t)
        self._hint = ""
        if t - self._t0 > self.cfg.total_timeout:
            self._fail("Took too long")
            return self.state(t)

        # --- presence ------------------------------------------------------
        if obs.num_faces == 0:
            if t - self._last_face_t > self.cfg.max_face_loss:
                self._fail("Face lost - stay in front of the camera")
            else:
                self._hint = "Keep your face in view"
            return self.state(t)
        if obs.num_faces > 1:
            self._multi_since = self._multi_since if self._multi_since is not None else t
            if t - self._multi_since > 1.5:
                self._fail("More than one face in view")
            else:
                self._hint = "Only one person should be in view"
            return self.state(t)
        self._multi_since = None
        self._last_face_t = t

        # --- same person throughout ---------------------------------------
        if self._track_id is not None and obs.track_id is not None and obs.track_id != self._track_id:
            self._need_reverify = True  # tracker lost the face: next embedding must match again
        if obs.track_id is not None:
            self._track_id = obs.track_id
        if obs.embedding is not None:
            emb = normalize(obs.embedding)
            frontal = abs(obs.yaw - self.base_yaw) <= self.cfg.center_degrees
            if self.base_embedding is None:
                if self.phase == "align" and abs(obs.yaw) <= self.cfg.center_degrees:
                    self.base_embedding = emb
                    self.embeddings.append(emb)
            elif abs(obs.yaw - self.base_yaw) <= self.cfg.continuity_max_yaw:
                # Embeddings of strongly turned faces legitimately differ from the frontal reference (real-world
                # pilot: genuine users failed at 31-49 deg), so continuity is checked near the baseline pose only;
                # during a turn the tracker keeps the identity (a lost track forces a re-check, see above).
                if float(emb @ self.base_embedding) < self.cfg.min_same_person:
                    self._fail("A different face appeared")
                    return self.state(t)
                if frontal:
                    self.embeddings.append(emb)
            self._need_reverify = False
        if self._need_reverify:
            self._hint = "Hold still..."
            return self.state(t)

        # --- phases ------------------------------------------------------------
        if self.phase == "align":
            self._update_align(obs)
        elif self.phase == "challenge":
            self._update_challenge(obs)
        elif self.phase == "recenter":
            if t - self._phase_t > self.cfg.challenge_timeout:
                self._fail("Please look back at the camera between actions")
            elif self._centered(obs):
                self._streak += 1
                if self._streak >= self.cfg.hold_frames:
                    self._set_phase("challenge", t)
        elif self.phase == "final":
            if t - self._phase_t > self.cfg.challenge_timeout:
                self._fail("Please look back at the camera")
            elif self._centered(obs):
                self._streak += 1
                if self._streak >= self.cfg.hold_frames and obs.embedding is not None:
                    self.phase = "passed"
        return self.state(t)

    def _update_align(self, obs: LivenessObservation) -> None:
        if abs(obs.yaw) > self.cfg.center_degrees:
            self._baseline.clear()
            self._baseline_landmarks.clear()
            self._hint = "Face the camera directly"
            return
        self._baseline.append((obs.yaw, obs.blink, obs.smile))
        if obs.landmarks is not None:
            self._baseline_landmarks.append(np.asarray(obs.landmarks, dtype=np.float64).reshape(5, 2))
        if len(self._baseline) >= self.cfg.baseline_frames and self.base_embedding is not None:
            yaws = [b[0] for b in self._baseline]
            blinks = [b[1] for b in self._baseline if b[1] is not None]
            smiles = [b[2] for b in self._baseline if b[2] is not None]
            self.base_yaw = float(np.median(yaws))
            self.base_blink = float(np.median(blinks)) if blinks else 0.0
            self.base_smile = float(np.median(smiles)) if smiles else 0.0
            if self._baseline_landmarks:
                self.base_landmarks = np.median(np.stack(self._baseline_landmarks), axis=0)
            self._set_phase("challenge", obs.t)

    def _onset(self, progress: float, elapsed: float) -> bool:
        """True while the current excursion started within the human response window after the prompt."""
        cfg = self.cfg
        if cfg.response_window is None:
            self._responded = True
            return True
        if self._prev_progress < 0.3 <= progress:  # an action starts
            self._onset_ok = cfg.min_response <= elapsed <= cfg.response_window
            self._responded = self._responded or self._onset_ok
        elif progress < 0.3:
            self._onset_ok = False
        self._prev_progress = progress
        return self._onset_ok

    def _parallax_progress(self, obs: LivenessObservation, sign: float) -> float | None:
        """Fraction of the required nose parallax reached (None when the parallax test does not apply)."""
        cfg = self.cfg
        if cfg.parallax_fraction is None or self.base_landmarks is None or obs.landmarks is None:
            return None
        need = cfg.parallax_fraction * cfg.turn_degrees
        return sign * parallax_yaw(self.base_landmarks, obs.landmarks) / need

    def _update_challenge(self, obs: LivenessObservation) -> None:
        c = self.sequence[self._i]
        cfg = self.cfg
        elapsed = obs.t - self._phase_t
        if elapsed > cfg.challenge_timeout:
            self._fail(f"Timed out waiting for: {c.instruction.lower()}")
            return
        sign = 1.0 if c is Challenge.TURN_LEFT else -1.0
        offset = obs.yaw - self.base_yaw
        smile_thr = max(0.50, self.base_smile + 0.30)

        # Order enforcement: an action belonging to a *different* challenge (turning the wrong way, turning during
        # a blink/smile prompt, smiling during another prompt) is a strike. One strike is forgiven (left/right
        # confusion with a mirrored preview is common); more fail the session. Hysteresis: a new strike needs the
        # face to return to neutral first, so one hesitant wrong turn near the threshold counts only once.
        wrong = (sign * offset <= -cfg.turn_degrees) if c.is_turn else (abs(offset) >= cfg.turn_degrees)
        if cfg.penalize_smiles and c is not Challenge.SMILE and obs.smile is not None:
            wrong = wrong or obs.smile >= smile_thr
        if wrong:
            self._wrong_streak += 1
            if self._wrong_streak >= cfg.hold_frames and not self._in_wrong:
                self._in_wrong = True
                self.strikes += 1
                if cfg.max_wrong_actions is not None and self.strikes > cfg.max_wrong_actions:
                    self._fail("Wrong action - follow the prompts in the order shown")
                    return
            self._hint = "Other way!" if (c.is_turn and sign * offset <= -cfg.turn_degrees) else "Only do what the prompt asks"
        else:
            self._wrong_streak = 0
            if abs(offset) < cfg.center_degrees and (obs.smile is None or obs.smile < self.base_smile + 0.15):
                self._in_wrong = False

        # Progress towards the requested action: 0 = at rest, 1 = done.
        if c.is_turn:
            progress = sign * offset / cfg.turn_degrees
            parallax = self._parallax_progress(obs, sign)
            if parallax is not None:  # a real head must also move its nose out of the eye/mouth plane
                progress = min(progress, parallax)
        elif c is Challenge.BLINK:
            if obs.blink is None:
                return
            closed_thr = max(cfg.blink_closed, self.base_blink + cfg.blink_closed_rel)
            progress = (obs.blink - self.base_blink) / (closed_thr - self.base_blink)
        else:
            if obs.smile is None:
                return
            progress = (obs.smile - self.base_smile) / (smile_thr - self.base_smile)

        # Response timing: the action must *start* like a human reaction to this prompt - not before
        # ``min_response`` (already under way when the prompt appeared, e.g. a looping video) and not after
        # ``response_window`` seconds.
        valid = self._onset(progress, elapsed)
        if cfg.response_window is not None and not self._responded and elapsed > cfg.response_window:
            self._fail("No response to the prompt in time")
            return
        self._meter = float(np.clip(progress, 0.0, 1.0)) if valid else 0.0

        done = False
        if c is Challenge.BLINK:
            if progress >= 1.0 and valid:
                self._blink_closed = True
            elif self._blink_closed and obs.blink <= max(cfg.blink_open, self.base_blink + cfg.blink_open_rel):
                done = True
        else:
            self._streak = self._streak + 1 if (progress >= 1.0 and valid) else 0
            done = self._streak >= cfg.hold_frames
        if done:
            self.completed.append(c)
            self._meter = 0.0
            self._i += 1
            self._set_phase("final" if self._i >= len(self.sequence) else "recenter", obs.t)


def observation_from_frame(result: FrameResult) -> LivenessObservation:
    """Build a liveness observation from an engine frame (primary face; tiny background faces ignored)."""
    primary = result.primary
    if primary is None:
        return LivenessObservation(result.t, 0)
    relevant = [f for f in result.faces if f.face_px >= 0.45 * primary.face_px]
    mesh = primary.mesh
    return LivenessObservation(
        t=result.t,
        num_faces=len(relevant),
        yaw=primary.pose.yaw,
        blink=mesh.blink if mesh else None,
        smile=mesh.smile if mesh else None,
        embedding=primary.embedding if primary.fresh else None,
        track_id=primary.track_id,
        landmarks=primary.landmarks,
    )


@dataclass
class VerificationResult:
    live: bool
    match: MatchResult
    reason: str
    duration: float
    challenges: list[Challenge]

    @property
    def verified(self) -> bool:
        return self.live and self.match.is_match


class VerificationTask:
    """Liveness challenges + identity decision ("face unlock" in one object)."""

    def __init__(self, gallery: Gallery, matcher: Matcher, config: LivenessConfig | None = None,
                 expressions_available: bool = False, require_owner: bool = False, rng: random.Random | None = None):
        self.gallery = gallery
        self.matcher = matcher
        self.require_owner = require_owner
        self.session = LivenessSession(config, expressions_available, rng)
        self.result: VerificationResult | None = None

    @property
    def needs_mesh(self) -> bool:
        return self.session.expressions

    def update(self, result: FrameResult) -> LivenessState:
        state = self.session.update(observation_from_frame(result))
        if state.finished and self.result is None:
            self.result = self._decide(state)
        return state

    def _decide(self, state: LivenessState) -> VerificationResult:
        s = self.session
        if state.phase != "passed" or not s.embeddings:
            return VerificationResult(False, MatchResult("pending"), state.reason or "Liveness failed", state.elapsed, list(s.completed))
        probe = normalize(np.mean(s.embeddings, axis=0))
        match = self.matcher.identify(self.gallery, probe)
        reason = ""
        if not match.is_match:
            reason = "Face not recognised" if self.gallery.persons else "Nobody is enrolled yet"
        elif self.require_owner and not (match.person and match.person.is_owner):
            match = MatchResult("uncertain", match.person, match.similarity, match.runner_up)
            reason = f"{match.person.name if match.person else 'This person'} is not the owner of this PC"
        return VerificationResult(True, match, reason, state.elapsed, list(s.completed))
