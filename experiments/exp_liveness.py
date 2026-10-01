"""Monte-Carlo evaluation of the challenge-response liveness protocol.

The production ``LivenessSession`` is driven at 20 fps by 5-point landmarks rendered from real 3-D face
geometry (300 MediaPipe meshes of LFW faces, ``exp_parallax.py``) through a pinhole webcam model
(f = 457 px, subject 40-70 cm away, 1 px landmark noise); yaw and nose parallax are computed from those
landmarks exactly as in the app. Blink / smile are simulated blendshape scores.

Presentations
  live users        head turns with random reaction time, amplitude, speed, retries, left/right confusion,
                    spontaneous blinks, sway
  flat print        a photo tilted back and forth by up to 70 deg (any planar motion)
  bent print        the photo curved into a cylinder (R = 60 / 40 / 30 mm) and tilted the same way
  replay            a recording of the genuine user completing one random sequence, looped on a screen
  universal replay  a recording of the genuine user cycling through every action
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from common import RESULTS, save_json

from faceguard.security.liveness import Challenge, LivenessConfig, LivenessObservation, LivenessSession, make_sequence
from faceguard.vision.geometry import pose_from_landmarks

FPS = 20.0
EMB_EVERY = 5
F_PX = 457.0
KEY = [468, 473, 1, 61, 291]
_SH = np.load(RESULTS / "face_shapes.npz")
# only the evaluation half: K and Kp were fitted on the disjoint calibration half (exp_parallax.py)
SHAPES = _SH["shapes"][_SH["eval_idx"]]


def forced_sequence(policy, cfg, expressions, rng):
    """Challenge sequence for a session: the protocol's own draw, or a fixed baseline policy."""
    if policy == "blink":
        return [Challenge.BLINK]
    return make_sequence(cfg.num_challenges, expressions, rng)


def unit(v):
    return v / np.linalg.norm(v)


class Subject:
    """One person (3-D face shape + identity embedding) at a random distance from the webcam."""

    def __init__(self, rng: np.random.Generator, noise_px: float = 1.0):
        self.rng = rng
        mesh = SHAPES[rng.integers(len(SHAPES))]
        self.head = mesh[KEY].copy()
        flat = self.head.copy()
        flat[:, 2] = 0.0
        self.flat = flat
        self.x2_mean = float(np.mean(mesh[:, 0] ** 2))
        self.dist = rng.uniform(400, 700)
        self.offset = np.array([rng.uniform(-40, 40), rng.uniform(-20, 20)])
        # people face the screen/camera: an off-centre head is turned slightly towards it (+ a habitual bias)
        self.face_yaw = math.degrees(math.atan2(-self.offset[0], self.dist)) + rng.normal(0, 2.0)
        self.noise = noise_px
        self.center = unit(rng.normal(size=128))

    def bent(self, radius):
        b = self.flat.copy()
        b[:, 2] = (b[:, 0] ** 2 - self.x2_mean) / (2.0 * radius)
        return b

    def embedding(self):
        return unit(self.center + 0.066 * self.rng.normal(size=128))

    def landmarks(self, pts3d, yaw_deg, pitch_deg=0.0, sway=(0.0, 0.0), facing=True):
        a, b = math.radians(yaw_deg + (self.face_yaw if facing else 0.0)), math.radians(pitch_deg)
        x, y, z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
        x, z = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
        y, z = y * math.cos(b) + z * math.sin(b), -y * math.sin(b) + z * math.cos(b)
        x = x + self.offset[0] + sway[0]
        y = y + self.offset[1] + sway[1]
        zz = self.dist + z
        lm = np.stack([F_PX * x / zz + 320.0, F_PX * y / zz + 240.0], axis=1)
        return lm + self.rng.normal(0, self.noise, lm.shape)


class LiveUser:
    """Follows the prompts; turns further / blinks harder when the progress meter does not fill."""

    def __init__(self, rng, p_wrong=0.05, react_median=0.7, amp_mean=30.0):
        self.rng, self.p_wrong = rng, p_wrong
        self.react_median, self.amp_mean = react_median, amp_mean
        self.base_blink, self.base_smile = rng.uniform(0.02, 0.20), rng.uniform(0.02, 0.25)
        self.instr, self.plan = None, {}
        self.next_blink = rng.exponential(3.5)
        self.drift = np.zeros(3)

    def _plan(self, t):
        r = self.rng
        self.plan = {"start": t + float(np.clip(r.lognormal(math.log(self.react_median), 0.3), 0.3, 2.5 * self.react_median / 0.7)),
                     "amp": float(np.clip(r.normal(self.amp_mean, 8), 8, 60)), "speed": r.uniform(60, 150),
                     "wrong": r.random() < self.p_wrong, "blink_dur": r.uniform(0.10, 0.35),
                     "blink_peak": r.uniform(0.40, 0.95), "smile_peak": r.uniform(0.45, 0.95)}

    def step(self, state, t):
        """-> (yaw_deg, pitch_deg, sway_mm, blink, smile) of the real head."""
        r = self.rng
        self.drift = 0.97 * self.drift + r.normal(0, [0.4, 0.3, 0.8])
        yaw, pitch = self.drift[0], self.drift[1]
        blink, smile = self.base_blink, self.base_smile
        if self.next_blink <= t < self.next_blink + 0.15:
            blink = 0.8
        elif t >= self.next_blink + 0.15:
            self.next_blink = t + r.exponential(3.5)
        if state is not None and state.phase == "challenge" and state.challenge is not None:
            c = state.challenge
            if state.instruction != self.instr:
                self.instr = state.instruction
                self._plan(t)
            d = t - self.plan["start"]
            if d > 0:
                if c.is_turn:
                    sign = 1.0 if c is Challenge.TURN_LEFT else -1.0
                    if self.plan["wrong"] and d < 1.2:
                        yaw += -sign * min(self.plan["amp"], d * self.plan["speed"])
                    else:
                        d2 = d - (1.2 if self.plan["wrong"] else 0.0)
                        amp = self.plan["amp"] + 8.0 * int(d2 // 2.0)
                        yaw += sign * min(amp, d2 * self.plan["speed"])
                elif c is Challenge.BLINK:
                    if (d % 1.5) < self.plan["blink_dur"]:
                        blink = min(1.0, self.plan["blink_peak"] + 0.1 * int(d // 1.5))
                elif c is Challenge.SMILE:
                    boost = 0.15 * int(d // 1.5)
                    smile = min(1.0, self.base_smile + min(1.0, d / 0.3) * (self.plan["smile_peak"] + boost - self.base_smile))
        else:
            self.instr = None
        return yaw, pitch, (self.drift[2], 0.0), blink, smile


def run(session, subject, frame_fn, max_t=60.0, record=None):
    """frame_fn(state, t) -> (points3d, yaw, pitch, sway, blink, smile)."""
    state, t, i = None, 0.0, 0
    rng = subject.rng
    while t < max_t:
        pts, yaw, pitch, sway, blink, smile = frame_fn(state, t)
        if record is not None:
            record.append((yaw, pitch, sway, blink, smile))
        lm = subject.landmarks(pts, yaw, pitch, sway)
        est = pose_from_landmarks(lm).yaw
        obs = LivenessObservation(
            t, 1, est,
            float(np.clip(blink + rng.normal(0, 0.03), 0, 1)) if session.expressions else None,
            float(np.clip(smile + rng.normal(0, 0.03), 0, 1)) if session.expressions else None,
            subject.embedding() if i % EMB_EVERY == 0 else None, 1, lm)
        state = session.update(obs)
        if state.finished:
            return state
        t += 1.0 / FPS
        i += 1
    return state


def live(n, cfg, expressions=True, noise_px=1.0, p_wrong=0.05, seed=0, react_median=0.7, amp_mean=30.0, policy=None):
    rng = np.random.default_rng(seed)
    ok, times, reasons = 0, [], Counter()
    for k in range(n):
        s = Subject(rng, noise_px)
        user = LiveUser(rng, p_wrong, react_median, amp_mean)
        srng = random.Random(seed * 1_000_003 + k)
        sess = LivenessSession(cfg, expressions, srng)
        if policy:
            sess.sequence = forced_sequence(policy, cfg, expressions, srng)
        st = run(sess, s, lambda state, t: (s.head, *user.step(state, t)))
        if st.phase == "passed":
            ok, _ = ok + 1, times.append(st.elapsed)
        else:
            reasons[st.reason.split(":")[0]] += 1
    t = np.array(times) if times else np.array([np.nan])
    return {"n": n, "frr": 1 - ok / n, "time_mean": float(np.nanmean(t)), "time_median": float(np.nanmedian(t)),
            "time_p95": float(np.nanpercentile(t, 95)), "fail_reasons": dict(reasons),
            "time_hist": np.histogram(t[np.isfinite(t)], bins=np.arange(0, 31, 1.0))[0].tolist()}


def print_attack(n, cfg, kind, seed=1, expressions=True, policy=None):
    """Adaptive attacker: reads each prompt and tilts the print towards the requested side, ever further."""
    rng = np.random.default_rng(seed)
    passed = 0
    for k in range(n):
        s = Subject(rng)
        pts = s.flat if kind == "flat" else s.bent(float(kind.split("R")[1]))
        max_tilt, speed = rng.uniform(40, 75), rng.uniform(20, 60)
        b0, s0 = rng.uniform(0.02, 0.3), rng.uniform(0.02, 0.4)
        prompt = {"text": None, "t": 0.0}

        def frame(state, t, pts=pts, max_tilt=max_tilt, speed=speed, b0=b0, s0=s0, prompt=prompt):
            tilt = 0.0
            if state is not None and state.phase == "challenge" and state.challenge is not None and state.challenge.is_turn:
                if state.instruction != prompt["text"]:
                    prompt["text"], prompt["t"] = state.instruction, t
                sign = 1.0 if state.challenge is Challenge.TURN_LEFT else -1.0
                tilt = sign * min(max_tilt, speed * (t - prompt["t"]))
            else:
                prompt["text"] = None
            jitter = (rng.normal(0, 3), rng.normal(0, 3))
            return pts, tilt + rng.normal(0, 1.5), rng.normal(0, 2), jitter, b0, s0

        srng = random.Random(9_000_011 * seed + k)
        sess = LivenessSession(cfg, expressions, srng)
        if policy:
            sess.sequence = forced_sequence(policy, cfg, expressions, srng)
        passed += run(sess, s, frame).phase == "passed"
    return {"n": n, "pass_rate": passed / n, "passed": passed}


def replay(n, cfg, expressions=True, seed=2, policy=None):
    rng = np.random.default_rng(seed)
    passed = done = 0
    rec_cfg = LivenessConfig(**{**cfg.__dict__, "max_wrong_actions": None})
    while done < n:
        s = Subject(rng)
        seq = forced_sequence(policy, cfg, expressions, random.Random(int(rng.integers(1 << 30))))
        user, trace = LiveUser(rng, 0.0), []
        rec = LivenessSession(rec_cfg, expressions, random.Random(0))
        rec.sequence = list(seq)
        if run(rec, s, lambda state, t: (s.head, *user.step(state, t)), record=trace).phase != "passed":
            continue
        off = int(rng.integers(len(trace)))

        def frame(state, t, trace=trace, off=off):
            yaw, pitch, sway, blink, smile = trace[(int(t * FPS) + off) % len(trace)]
            return s.head, yaw, pitch, sway, blink, smile  # the screen shows the recorded 3-D head

        srng = random.Random(int(rng.integers(1 << 30)))
        sess = LivenessSession(cfg, expressions, srng)
        if policy:
            sess.sequence = forced_sequence(policy, cfg, expressions, srng)
        passed += run(sess, s, frame).phase == "passed"
        done += 1
    return {"n": n, "pass_rate": passed / n, "passed": passed}


def universal(n, cfg, seed=3, expressions=True, policy=None):
    rng = np.random.default_rng(seed)
    passed = 0
    for k in range(n):
        s = Subject(rng)
        period, off, order = rng.uniform(1.4, 2.6), rng.uniform(0, 10), rng.permutation(4)

        def frame(state, t, period=period, off=off, order=order):
            kk, ph = divmod(t + off, period)
            act = int(order[int(kk) % 4])
            doing = ph < 0.6 * period
            yaw = 30.0 if (doing and act == 0) else (-30.0 if (doing and act == 1) else 0.0)
            blink = 0.85 if (doing and act == 2 and ph < 0.2) else 0.05
            smile = 0.8 if (doing and act == 3) else 0.1
            return s.head, yaw, 0.0, (0.0, 0.0), blink, smile

        srng = random.Random(77_777 * seed + k)
        sess = LivenessSession(cfg, expressions, srng)
        if policy:
            sess.sequence = forced_sequence(policy, cfg, expressions, srng)
        passed += run(sess, s, frame).phase == "passed"
    return {"n": n, "pass_rate": passed / n, "passed": passed}


def sequence_stats(n_challenges, expressions, draws=200_000):
    rng = random.Random(5)
    counts = Counter(tuple(c.value for c in make_sequence(n_challenges, expressions, rng)) for _ in range(draws))
    p = np.array(list(counts.values()), float) / draws
    return {"distinct": len(counts), "max_prob": float(p.max()), "min_entropy_bits": float(-math.log2(p.max())),
            "shannon_bits": float(-(p * np.log2(p)).sum())}


FUNCS = {"live": live, "print": print_attack, "replay": replay, "universal": universal}


def _task(spec):
    group, key, fn, args, kwargs = spec
    return group, key, FUNCS[fn](*args, **kwargs)


def main():
    from concurrent.futures import ProcessPoolExecutor

    full = LivenessConfig()
    no_parallax = LivenessConfig(parallax_fraction=None)
    no_order = LivenessConfig(max_wrong_actions=None)
    no_smile = LivenessConfig(penalize_smiles=False)
    no_timing = LivenessConfig(response_window=None)
    short_window = LivenessConfig(response_window=2.5)
    long_window = LivenessConfig(response_window=5.0)
    # baselines under the same simulation (Section V-E): conventional schemes without our defenses
    yaw_turns = LivenessConfig(parallax_fraction=None, max_wrong_actions=None, response_window=None)
    conventional = LivenessConfig(parallax_fraction=None, max_wrong_actions=None, response_window=None, penalize_smiles=False)
    blink_cfg = LivenessConfig(num_challenges=1, max_wrong_actions=None, response_window=None)
    baselines = {"yaw_turns": (yaw_turns, False, None), "blink_only": (blink_cfg, True, "blink"),
                 "conventional": (conventional, True, None), "faceguard": (full, True, None)}

    T = []  # (group, key, function, args, kwargs); every seed is fixed and listed here
    T += [("live", "full", "live", (2000, full), {}),
          ("live", "no_parallax", "live", (1000, no_parallax), {"seed": 11}),
          ("live", "no_order_enforcement", "live", (1000, no_order), {"seed": 12}),
          ("live", "no_smile_rule", "live", (1000, no_smile), {"seed": 16}),
          ("live", "no_timing", "live", (1000, no_timing), {"seed": 17}),
          ("live", "window_2_5", "live", (1000, short_window), {"seed": 18}),
          ("live", "window_5_0", "live", (1000, long_window), {"seed": 19}),
          ("live", "turns_only", "live", (1000, full), {"expressions": False, "seed": 13}),
          ("live", "noise_2px", "live", (1000, full), {"noise_px": 2.0, "seed": 14}),
          ("live", "no_confusion", "live", (1000, full), {"p_wrong": 0.0, "seed": 15})]
    # behavioural-model sensitivity (reaction delay, turn amplitude, left/right confusion)
    for m in (0.5, 1.0, 1.4):
        T.append(("sens", f"react_{m}", "live", (1000, full), {"react_median": m, "seed": 60 + int(m * 10)}))
    for a in (22.0, 38.0):
        T.append(("sens", f"amp_{int(a)}", "live", (1000, full), {"amp_mean": a, "seed": 80 + int(a)}))
    for p in (0.10, 0.20):
        T.append(("sens", f"confusion_{p}", "live", (1000, full), {"p_wrong": p, "seed": 120 + int(p * 100)}))
    for tau in (12, 15, 18, 21, 24, 27):
        for noise in (0.5, 1.0, 2.0):
            T.append(("sweep", f"{tau}/{noise}", "live", (400, LivenessConfig(turn_degrees=float(tau))),
                      {"expressions": False, "noise_px": noise, "seed": int(tau * 7 + noise * 3)}))
    for kind in ("flat", "bent_R60", "bent_R40", "bent_R30"):
        T += [("att", f"{kind}/full", "print", (1000, full, kind), {}),
              ("att", f"{kind}/turns_only", "print", (1000, full, kind), {"seed": 22, "expressions": False}),
              ("att", f"{kind}/turns_only_no_parallax", "print", (1000, no_parallax, kind), {"seed": 23, "expressions": False})]
    T += [("att", "replay/full", "replay", (800, full), {}),
          ("att", "replay/no_timing", "replay", (800, no_timing), {"seed": 34}),
          ("att", "replay/no_smile_rule", "replay", (800, no_smile), {"seed": 33}),
          ("att", "replay/no_order_enforcement", "replay", (800, no_order), {"seed": 31}),
          ("att", "replay/turns_only", "replay", (800, full), {"expressions": False, "seed": 32}),
          ("att", "replay/window_2_5", "replay", (800, short_window), {"seed": 35}),
          ("att", "replay/window_5_0", "replay", (800, long_window), {"seed": 36}),
          ("att", "universal/full", "universal", (1500, full), {}),
          ("att", "universal/no_timing", "universal", (1500, no_timing), {"seed": 43}),
          ("att", "universal/no_smile_rule", "universal", (1500, no_smile), {"seed": 42}),
          ("att", "universal/no_order_enforcement", "universal", (1500, no_order), {"seed": 41}),
          ("att", "universal/window_2_5", "universal", (1500, short_window), {"seed": 44}),
          ("att", "universal/window_5_0", "universal", (1500, long_window), {"seed": 45}),
          ("att", "universal/no_order_no_timing", "universal", (1500, LivenessConfig(max_wrong_actions=None, response_window=None)),
           {"seed": 46})]
    for name, (cfg, expr, pol) in baselines.items():
        if name == "faceguard":
            continue  # reuses the full-protocol rows above
        T += [("base", f"{name}/live", "live", (1000, cfg), {"expressions": expr, "policy": pol, "seed": 200}),
              ("base", f"{name}/flat", "print", (1000, cfg, "flat"), {"expressions": expr, "policy": pol, "seed": 201}),
              ("base", f"{name}/bent_R40", "print", (1000, cfg, "bent_R40"), {"expressions": expr, "policy": pol, "seed": 202}),
              ("base", f"{name}/replay", "replay", (800, cfg), {"expressions": expr, "policy": pol, "seed": 203}),
              ("base", f"{name}/universal", "universal", (1500, cfg), {"expressions": expr, "policy": pol, "seed": 204})]

    out = {"fps": FPS, "focal_px": F_PX, "shapes": int(len(SHAPES)), "config": full.__dict__,
           "seeds": {f"{g}/{k}": kw.get("seed", {"live": 0, "print": 1, "replay": 2, "universal": 3}[fn]) for g, k, fn, _, kw in T}}
    out["sequences"] = {f"n={n},expr={e}": sequence_stats(n, e) for n in (2, 3, 4) for e in (True, False)}
    res = {}
    # longest tasks first keeps the pool busy until the end
    order = sorted(T, key=lambda s: -s[3][0] * (3 if s[2] == "universal" else 2 if s[2] == "replay" else 1))
    with ProcessPoolExecutor(8) as pool:
        for group, key, r in pool.map(_task, order):
            res[(group, key)] = r
            print(f"  {group:6s} {key:34s} " + (f"FRR {r['frr']:.4f}" if "frr" in r else f"pass {r['pass_rate']:.4f}"), flush=True)
    out["live"] = {k: v for (g, k), v in res.items() if g == "live"}
    out["sensitivity"] = {k: v for (g, k), v in res.items() if g == "sens"}
    out["tau_sweep"] = [{"tau": int(k.split("/")[0]), "noise_px": float(k.split("/")[1]), "frr": v["frr"], "n": v["n"],
                         "time_mean": v["time_mean"]} for (g, k), v in res.items() if g == "sweep"]
    out["tau_sweep"].sort(key=lambda s: (s["tau"], s["noise_px"]))
    out["attacks"] = {k: v for (g, k), v in res.items() if g == "att"}
    base = {k: v for (g, k), v in res.items() if g == "base"}
    for kind, src in (("live", "live/full"), ("flat", "att/flat/full"), ("bent_R40", "att/bent_R40/full"),
                      ("replay", "att/replay/full"), ("universal", "att/universal/full")):
        g, k = src.split("/", 1)
        base[f"faceguard/{kind}"] = res[(g, k)]
    out["baselines"] = base
    save_json("liveness", out)


if __name__ == "__main__":
    main()
