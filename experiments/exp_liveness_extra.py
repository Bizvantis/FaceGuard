"""Additional Monte-Carlo analyses of the liveness protocol (reviewer revision), using exp_liveness.py unchanged.

A  Independent repetitions. The principal configurations are re-run with five new master seeds (1001-1005). For each
   configuration we report the mean and standard deviation over seeds and a 95% interval from a cluster bootstrap over
   the 150 evaluation face meshes (sessions that share a face are not independent), pooled over the five seeds.
B  Adaptive replay. The attacker owns several recordings (clips) of the genuine user performing each action:
     prompt_follow_human  an operator watches the prompts and starts the matching clip after a human reaction delay
                          (log-normal, median 0.8 s); clip speed x0.8-1.25; 5 clips per action
     prompt_follow_auto   the same, driven by software that reads the prompt (delay 0.20-0.45 s)
     blind_edit           a pre-edited video: clips of three distinct actions in a random order with random gaps and
                          speeds, looped from a random start; the attacker does not see the prompts
C  Threshold sensitivity. Flat/bent prints (turn-only, parallax gate on) and live users under parallax fractions
   0.6-1.0 of the 18 deg turn threshold and under turn thresholds of 15 and 21 deg.
D  Landmark-noise sweep (1-4 px) for prints and live users, used to compare the simulated noise model with the
   parallax measured in the real-world phone attacks (exp_parallax_real.py).
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict

import numpy as np

import exp_liveness as L
from common import save_json

from faceguard.security.liveness import Challenge, LivenessConfig, LivenessSession

SEEDS = (1001, 1002, 1003, 1004, 1005)
HEADS = L.SHAPES[:, L.KEY]

# ------------------------------------------------------------------ per-session bookkeeping (for the cluster bootstrap)
_LOG: list[tuple[int, bool]] = []
_LAST = {"face": -1}
_orig_init, _orig_run = L.Subject.__init__, L.run


def _init(self, rng, noise_px=1.0):
    _orig_init(self, rng, noise_px)
    _LAST["face"] = int(np.argmin(np.abs(HEADS - self.head).sum(axis=(1, 2))))


def _run(session, subject, frame_fn, max_t=60.0, record=None):
    st = _orig_run(session, subject, frame_fn, max_t, record)
    if record is None:  # recording runs of the replay attacker are not attack attempts
        _LOG.append((_LAST["face"], st.phase == "passed"))
    return st


L.Subject.__init__ = _init
L.run = _run


def print_attack(n, cfg, kind, seed=1, expressions=True, noise_px=1.0):
    """exp_liveness.print_attack with a configurable landmark noise."""
    old = L.Subject.__init__.__defaults__
    L.Subject.__init__.__defaults__ = (noise_px,)
    try:
        return L.print_attack(n, cfg, kind, seed=seed, expressions=expressions)
    finally:
        L.Subject.__init__.__defaults__ = old


# ------------------------------------------------------------------ adaptive replay
def make_clip(action, rng, speed):
    """Frames (yaw, blink, smile) at 20 fps of the genuine user performing one action, played at a speed factor."""
    n = int(round(2.4 * L.FPS / speed))
    tt = np.arange(n) / L.FPS * speed  # clip time
    yaw, blink, smile = np.zeros(n), np.full(n, 0.05), np.full(n, 0.10)
    if action in (Challenge.TURN_LEFT, Challenge.TURN_RIGHT):
        amp, vel = float(np.clip(rng.normal(30, 8), 12, 55)), rng.uniform(60, 150)
        sign = 1.0 if action is Challenge.TURN_LEFT else -1.0
        up = np.minimum(amp, vel * tt)
        down = np.maximum(0.0, amp - vel * np.maximum(0.0, tt - (amp / vel + 0.6)))
        yaw = sign * np.minimum(up, down)
    elif action is Challenge.BLINK:
        dur, peak = rng.uniform(0.10, 0.35), rng.uniform(0.40, 0.95)
        blink = np.where((tt > 0.2) & (tt < 0.2 + dur), peak, 0.05)
    else:
        peak = rng.uniform(0.45, 0.95)
        smile = 0.10 + (peak - 0.10) * np.clip(np.minimum(tt / 0.3, (2.0 - tt) / 0.3), 0, 1)
    return list(zip(yaw, blink, smile))


def adaptive_replay(n, cfg, mode, seed, expressions=True, clips_per_action=5):
    rng = np.random.default_rng(seed)
    passed = 0
    actions = [Challenge.TURN_LEFT, Challenge.TURN_RIGHT, Challenge.BLINK, Challenge.SMILE]
    for k in range(n):
        s = L.Subject(rng)
        lib = {a: [make_clip(a, rng, rng.uniform(0.8, 1.25)) for _ in range(clips_per_action)] for a in actions}
        play = {"frames": [], "start": None, "prompt": None}
        if mode == "blind_edit":
            order = [actions[i] for i in rng.permutation(4)[:3]]
            video = []
            for a in order:
                video += [(0.0, 0.05, 0.10)] * int(rng.uniform(0.6, 2.5) * L.FPS)
                video += lib[a][int(rng.integers(clips_per_action))]
            off = int(rng.integers(len(video)))

        def frame(state, t, play=play):
            if mode == "blind_edit":
                yaw, blink, smile = video[(int(t * L.FPS) + off) % len(video)]
                return s.head, yaw, 0.0, (0.0, 0.0), blink, smile
            if state is not None and state.phase == "challenge" and state.challenge is not None \
                    and state.instruction != play["prompt"]:
                play["prompt"] = state.instruction
                delay = (rng.lognormal(math.log(0.8), 0.3) if mode == "prompt_follow_human" else rng.uniform(0.20, 0.45))
                play["frames"] = lib[state.challenge][int(rng.integers(clips_per_action))]
                play["start"] = t + delay
            if play["start"] is not None and t >= play["start"]:
                i = int((t - play["start"]) * L.FPS)
                if i < len(play["frames"]):
                    yaw, blink, smile = play["frames"][i]
                    return s.head, yaw, 0.0, (0.0, 0.0), blink, smile
            return s.head, 0.0, 0.0, (0.0, 0.0), 0.05, 0.10

        sess = LivenessSession(cfg, expressions, random.Random(int(rng.integers(1 << 30))))
        passed += L.run(sess, s, frame).phase == "passed"
    return {"n": n, "pass_rate": passed / n, "passed": passed}


# ------------------------------------------------------------------ tasks
def _task(spec):
    group, key, fn, args, kwargs = spec
    _LOG.clear()
    if fn == "live":
        r = L.live(*args, **kwargs)
    elif fn == "print":
        r = print_attack(*args, **kwargs)
    elif fn == "replay":
        r = L.replay(*args, **kwargs)
    elif fn == "universal":
        r = L.universal(*args, **kwargs)
    else:
        r = adaptive_replay(*args, **kwargs)
    r["sessions"] = list(_LOG)
    return group, key, r


def cluster_ci(sessions, success, b=2000, seed=7):
    """95% interval of a rate from a bootstrap over face meshes (the cluster), plus the point estimate."""
    by = defaultdict(lambda: [0, 0])
    for face, ok in sessions:
        by[face][0] += 1
        by[face][1] += int(ok == success)
    faces = np.array(list(by.values()), np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(faces), (b, len(faces)))
    sums = faces[idx].sum(axis=1)
    rates = sums[:, 1] / sums[:, 0]
    return float(faces[:, 1].sum() / faces[:, 0].sum()), np.percentile(rates, [2.5, 97.5]).tolist(), int(len(faces))


def main():
    from concurrent.futures import ProcessPoolExecutor

    full = LivenessConfig()
    no_parallax = LivenessConfig(parallax_fraction=None)
    no_order_no_timing = LivenessConfig(max_wrong_actions=None, response_window=None)
    T = []
    principal = [("live/full", "live", (1000, full), {}),
                 ("live/turns_only", "live", (1000, full), {"expressions": False}),
                 ("flat/full", "print", (1000, full, "flat"), {}),
                 ("bent_R40/full", "print", (1000, full, "bent_R40"), {}),
                 ("flat/turns_only", "print", (1000, full, "flat"), {"expressions": False}),
                 ("flat/turns_only_no_parallax", "print", (1000, no_parallax, "flat"), {"expressions": False}),
                 ("bent_R40/turns_only", "print", (1000, full, "bent_R40"), {"expressions": False}),
                 ("replay/full", "replay", (800, full), {}),
                 ("universal/full", "universal", (1500, full), {}),
                 ("universal/no_order_no_timing", "universal", (1500, no_order_no_timing), {})]
    for j, sd in enumerate(SEEDS):
        for i, (key, fn, args, kw) in enumerate(principal):
            T.append(("seeds", f"{key}#{sd}", fn, args, {**kw, "seed": sd * 100 + i}))
    for mode in ("prompt_follow_human", "prompt_follow_auto", "blind_edit"):
        T.append(("adaptive", f"{mode}/full", "adaptive", (1000, full, mode), {"seed": 500 + len(T)}))
        T.append(("adaptive", f"{mode}/turns_only", "adaptive", (1000, full, mode), {"seed": 500 + len(T), "expressions": False}))
    for frac in (0.6, 0.7, 0.8, 0.9, 1.0):
        cfg = LivenessConfig(parallax_fraction=frac)
        T += [("thr", f"frac_{frac}/flat", "print", (1000, cfg, "flat"), {"expressions": False, "seed": 700 + int(frac * 10)}),
              ("thr", f"frac_{frac}/bent_R40", "print", (1000, cfg, "bent_R40"), {"expressions": False, "seed": 720 + int(frac * 10)}),
              ("thr", f"frac_{frac}/live_turns", "live", (1000, cfg), {"expressions": False, "seed": 740 + int(frac * 10)}),
              ("thr", f"frac_{frac}/live_full", "live", (1000, cfg), {"seed": 760 + int(frac * 10)})]
    for tau in (15.0, 21.0):
        cfg = LivenessConfig(turn_degrees=tau)
        T += [("thr", f"tau_{int(tau)}/flat", "print", (1000, cfg, "flat"), {"expressions": False, "seed": 800 + int(tau)}),
              ("thr", f"tau_{int(tau)}/bent_R40", "print", (1000, cfg, "bent_R40"), {"expressions": False, "seed": 830 + int(tau)}),
              ("thr", f"tau_{int(tau)}/live_turns", "live", (1000, cfg), {"expressions": False, "seed": 860 + int(tau)}),
              ("thr", f"tau_{int(tau)}/live_full", "live", (1000, cfg), {"seed": 890 + int(tau)})]
    for noise in (1.0, 2.0, 3.0, 4.0):
        T += [("noise", f"{noise}/flat_turns", "print", (1000, full, "flat"), {"expressions": False, "noise_px": noise, "seed": 900 + int(noise)}),
              ("noise", f"{noise}/flat_turns_no_parallax", "print", (1000, no_parallax, "flat"),
               {"expressions": False, "noise_px": noise, "seed": 910 + int(noise)}),
              ("noise", f"{noise}/live_turns", "live", (1000, full), {"expressions": False, "noise_px": noise, "seed": 920 + int(noise)}),
              ("noise", f"{noise}/live_full", "live", (1000, full), {"noise_px": noise, "seed": 930 + int(noise)})]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    res = {}
    order = sorted(T, key=lambda s: -s[3][0] * (3 if s[2] in ("universal", "replay") else 1))
    with ProcessPoolExecutor(workers) as pool:
        for group, key, r in pool.map(_task, order):
            res[(group, key)] = r
            print(f"  {group:8s} {key:42s} " + (f"FRR {r['frr']:.4f}" if "frr" in r else f"pass {r['pass_rate']:.4f}"), flush=True)
    out = {"seeds": SEEDS, "faces": int(len(HEADS)), "repeated": {}, "adaptive": {}, "thresholds": {}, "noise": {}}
    for key, fn, _, _ in principal:
        runs = [res[("seeds", f"{key}#{sd}")] for sd in SEEDS]
        live = fn == "live"
        vals = np.array([r["frr"] if live else r["pass_rate"] for r in runs])
        sessions = [x for r in runs for x in r["sessions"]]
        point, ci, faces = cluster_ci(sessions, success=not live)
        out["repeated"][key] = {"metric": "BPCER" if live else "APCER", "per_seed": vals.tolist(), "mean": float(vals.mean()),
                                "sd": float(vals.std(ddof=1)), "pooled": point, "cluster_ci": ci, "faces": faces,
                                "sessions": len(sessions)}
    for (g, k), r in res.items():
        r = {kk: v for kk, v in r.items() if kk != "sessions"}
        if g == "adaptive":
            out["adaptive"][k] = r
        elif g == "thr":
            out["thresholds"][k] = r
        elif g == "noise":
            out["noise"][k] = r
    save_json("liveness_extra", out)


if __name__ == "__main__":
    main()
