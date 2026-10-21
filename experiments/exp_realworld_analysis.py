"""Summarize the real-world evaluation (results/realworld/*.jsonl) into results/realworld.json.

The evaluation ran in two rounds with the same test program. Round 1 (pilot_trials_protocol_v1.jsonl) used the
earlier liveness protocol and exposed the faults that were then corrected; round 2 (trials.jsonl) used the
corrected protocol. Both rounds are summarized: genuine-user rates are reported per round (the protocol differs),
and the phone-screen attacks of both rounds are also pooled. Round-2 attack trials that the participant marked as
not clean (own face visible) were discarded at acquisition time; round 1 had no such check, and its one affected
trial is listed in INVALID_ROUND1 with the evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import RESULTS, save_json

R = RESULTS / "realworld"
MIN_PERSISTENCE = {"guard_shoulder": 3.0}  # seconds an onlooker must be in view before the guard alerts
# Round-1 attack trials that were not clean presentations (no clean-presentation check existed in round 1).
INVALID_ROUND1 = {
    "81f6189124": "participant reported their own face in view; the session completed a smile and a blink with "
                  "64 deg of parallax, which a phone photo cannot produce",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def load(name):
    p = R / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def trace(r):
    return json.loads((R / r["trace"]).read_text()) if r.get("trace") else []


def turn_stats(rows):
    """Max |yaw change| and max |parallax| reached during turn challenges, per trial."""
    yaws, pars = [], []
    for r in rows:
        tr = trace(r)
        base = [f["yaw"] for f in tr if f["phase"] == "align" and f["yaw"] is not None]
        b = float(np.median(base)) if base else 0.0
        seg = [f for f in tr if f["phase"] == "challenge" and f["challenge"] in ("turn_left", "turn_right") and f["yaw"] is not None]
        if seg:
            yaws.append(max(abs(f["yaw"] - b) for f in seg))
            pp = [abs(f["parallax"]) for f in seg if f["parallax"] is not None]
            if pp:
                pars.append(max(pp))
    q = lambda v: {"n": len(v), "median": float(np.median(v)) if v else None, "max": float(max(v)) if v else None}  # noqa: E731
    return {"max_yaw_change": q(yaws), "max_parallax": q(pars)}


def unlock_summary(rows, attack=False):
    n = len(rows)
    k = sum(r["verified"] for r in rows)
    out = {"n": n, "verified": k, "rate": k / n if n else None, "ci": wilson(k, n),
           "reasons": dict(Counter(r["reason"] or "passed" for r in rows)),
           "reached_challenge": sum(r.get("reached_challenge", True) for r in rows),
           "duration_mean": float(np.mean([r["duration"] for r in rows if r["verified"]])) if k else None,
           "fps_median": float(np.median([r["fps"] for r in rows if r.get("fps")])) if any(r.get("fps") for r in rows) else None}
    if not attack:
        sims = [r["similarity"] for r in rows if r["live"] and r["similarity"] is not None]
        out["live_rate"] = sum(r["live"] for r in rows) / n if n else None
        out["live_ci"] = wilson(sum(r["live"] for r in rows), n)
        out["similarity_median"] = float(np.median(sims)) if sims else None
        out["similarity_min"] = float(min(sims)) if sims else None
    return out


def guard_summary(rows, kind):
    if not rows:
        return None
    if kind == "guard_desk":
        hours = sum(r["duration_s"] for r in rows) / 3600
        labels = Counter()
        for r in rows:
            labels.update(r["labels"])
        tot = sum(labels.values())
        return {"sessions": len(rows), "hours": hours, "frames": tot,
                "label_fraction": {k: v / tot for k, v in labels.items()},
                "intruder_locks": sum(r["intruder_locks"] for r in rows), "absence_locks": sum(r["absence_locks"] for r in rows),
                "shoulder_alerts": sum(r["shoulder_alerts"] for r in rows),
                "longest_gap_s": max(r["longest_gap_s"] for r in rows)}
    # time from the beep (owner leaves / volunteer starts) to the decision, independent of which face was seen first
    def since(r):
        if not r["detected"] or r["latency_s"] is None or r["t_event_s"] is None:
            return None
        return r["t_reference_s"] + r["latency_s"] - r["t_event_s"]
    # The shoulder-surfer rule needs an onlooker in view for MIN_PERSISTENCE seconds (guard.py shoulder_seconds), so an
    # alert sooner than that after the cue means the volunteer was already in view before the cue: such a trial still
    # counts as a detection but cannot time the response.
    minimum = MIN_PERSISTENCE.get(kind, 0.0)
    timed = [r for r in rows if since(r) is not None and since(r) >= minimum]
    lat = [r["latency_s"] for r in timed]
    s = [since(r) for r in timed]
    return {"trials": len(rows), "detected": sum(r["detected"] for r in rows), "detected_ci": wilson(sum(r["detected"] for r in rows), len(rows)),
            "timed_trials": len(timed), "untimed_trials": sum(r["detected"] for r in rows) - len(timed),
            "latency_median": float(np.median(lat)) if lat else None, "latency_max": float(max(lat)) if lat else None,
            "latency_min": float(min(lat)) if lat else None,
            "since_event_median": float(np.median(s)) if s else None,
            "subjects": sorted({r["subject"] for r in rows})}


def main():
    v1, v2 = load("pilot_trials_protocol_v1.jsonl"), load("trials.jsonl")
    blocks = lambda rows, key, subj=None: [r for r in rows if r["block"] == key and (subj is None or r["subject"] == subj)]  # noqa: E731
    out = {"pilot_v1": {"trials": len(v1), "by_block": dict(Counter(r["block"] for r in v1)),
                        "owner_live": unlock_summary(blocks(v1, "owner_live") + blocks(v1, "owner_turns")),
                        "reasons": dict(Counter(f"{r['block']}: {r['reason'] or 'passed'}" for r in v1
                                                if r["block"].startswith("owner")))}}
    ev = {}
    for key in ("owner_live", "owner_turns"):
        rows = blocks(v2, key, "owner")
        if rows:
            ev[key] = unlock_summary(rows)
            ev[key].update(turn_stats(rows))
    for key in ("phone_full", "phone_turns", "phone_turns_noparallax"):
        rows = [r for r in blocks(v2, key, "owner") if r.get("clean_presentation", True)]
        if rows:
            ev[key] = unlock_summary(rows, attack=True)
            ev[key].update(turn_stats(rows))
    vol = blocks(v2, "volunteer_live")
    if vol:
        s = unlock_summary(vol)
        s["false_accepts"] = sum(r["verified"] for r in vol)
        s["matched_owner"] = sum(r["matched_owner"] for r in vol)
        s["far_ci"] = wilson(s["false_accepts"], len(vol))
        s["subjects"] = sorted({r["subject"] for r in vol})
        s["max_similarity_to_owner"] = max((r["similarity"] or 0.0) for r in vol)
        ev["volunteer_live"] = s
    for key in ("guard_desk", "guard_walkaway", "guard_intruder", "guard_shoulder", "guard_lockwatch"):
        s = guard_summary(blocks(v2, key), key)
        if s:
            ev[key] = s
    live_all = blocks(v2, "owner_live", "owner") + blocks(v2, "owner_turns", "owner") + vol
    if live_all:
        k = sum(not r["live"] for r in live_all)
        ev["bpcer_all_live"] = {"n": len(live_all), "rejected": k, "rate": k / len(live_all), "ci": wilson(k, len(live_all)),
                                "subjects": sorted({r["subject"] for r in live_all})}
    out["evaluation_v2"] = ev
    # round-1 attacks and all phone-screen attacks of both rounds
    att1 = [r for r in blocks(v1, "phone_full", "owner") if r["id"] not in INVALID_ROUND1]
    s1 = unlock_summary(att1, attack=True)
    s1.update(turn_stats(att1))
    out["round1_attacks"] = s1
    att2 = [r for k in ("phone_full", "phone_turns", "phone_turns_noparallax") for r in blocks(v2, k, "owner")
            if r.get("clean_presentation", True)]
    pooled = att1 + att2
    k = sum(r["verified"] for r in pooled)
    ts = turn_stats(pooled)
    over = []  # attacks whose tilted phone exceeded the parallax threshold (0.8 x 18 deg) during a turn challenge
    for rnd, rows in (("round1", att1), ("round2", att2)):
        for r in rows:
            m = turn_stats([r])["max_parallax"]["max"]
            if m is not None and m > 14.4:
                over.append({"round": rnd, "id": r["id"], "max_parallax": m, "reason": r["reason"]})
    out["all_rounds"] = {"sessions_recorded": len(v1) + len(v2), "sessions_discarded": len(INVALID_ROUND1),
                         "discarded": INVALID_ROUND1, "attacks": len(pooled), "attacks_accepted": k,
                         "attack_ci": wilson(k, len(pooled)), "attack_reasons": dict(Counter(r["reason"] or "passed" for r in pooled)),
                         "attacks_over_parallax_threshold": over, **ts}
    own2 = blocks(v2, "owner_live", "owner") + blocks(v2, "owner_turns", "owner")
    out["evaluation_v2"]["owner_reasons"] = dict(Counter(r["reason"] or "passed" for r in own2))
    save_json("realworld", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
