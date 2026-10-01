"""Summarize the real-world pilot study (results/realworld/*.jsonl) into results/realworld.json.

Protocol v1 trials (pilot_trials_protocol_v1.jsonl) are reported only as the pilot that motivated the fixes;
all evaluation numbers come from protocol v2 trials (trials.jsonl). Attack trials that the participant marked
as not clean (own face visible) were discarded at acquisition time.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import RESULTS, save_json

R = RESULTS / "realworld"


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
    lat = [r["latency_s"] for r in rows if r["detected"] and r["latency_s"] is not None]
    # time from the beep (owner leaves / volunteer starts) to the decision, independent of which face was seen first
    since = [r["t_reference_s"] + r["latency_s"] - r["t_event_s"] for r in rows
             if r["detected"] and r["latency_s"] is not None and r["t_event_s"] is not None]
    return {"trials": len(rows), "detected": sum(r["detected"] for r in rows), "detected_ci": wilson(sum(r["detected"] for r in rows), len(rows)),
            "latency_median": float(np.median(lat)) if lat else None, "latency_max": float(max(lat)) if lat else None,
            "latency_min": float(min(lat)) if lat else None,
            "since_event_median": float(np.median(since)) if since else None,
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
    save_json("realworld", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
