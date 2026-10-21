"""Analysis of the physical study (exp_physical.py) -> results/physical.json (anonymous summary).

    ..\\.venv\\Scripts\\python.exe exp_physical_analysis.py --freeze   (after the owner's calibration sessions)
    ..\\.venv\\Scripts\\python.exe exp_physical_analysis.py            (full analysis)

Freezing rule for the tracked-feature (flow) gate, fixed before any attack was recorded and using only the owner's
bona fide head turns: for every turn challenge, take the planarity residual at the first frame at which the
yaw ratio reaches the 18 deg turn threshold; the gate threshold is 0.8 x the median of these values - the same
"80 % of a real head" rule as the five-landmark parallax gate (0.8 x 18 deg). The volunteers are held-out users.

Units: a bona fide or attack *session* for APCER/BPCER; a *turn challenge* for the paired gate analysis (the same
attempt judged with and without each gate); an *event* or an *hour of monitoring* for the guard.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

from common import RESULTS, save_json

OUT = RESULTS / "physical"
LOG = OUT / "trials.jsonl"
FROZEN = OUT / "frozen_thresholds.json"
TURN, PAR_NEED, HOLD = 18.0, 0.8 * 18.0, 2
FLOW_INDEX = {"n": 0, "rms": 1, "p75": 2, "outlier_frac": 3, "shift": 4}
PAD_NAMES = ["minifasnet", "mn3", "cvpr24_r50", "cvpr24_swin"]


# ------------------------------------------------------------------ loading
def load():
    recs = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()] if LOG.exists() else []
    for r in recs:
        r["_trace"] = json.loads((OUT / r["trace"]).read_text(encoding="utf-8")) if r.get("trace") else None
    return recs


def cp_interval(k, n, alpha=0.05):
    """Clopper-Pearson interval for k successes in n trials (as fractions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def rate(k, n):
    lo, hi = cp_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "lo": lo, "hi": hi}


def poisson_rate(k, hours):
    lo = 0.0 if k == 0 else stats.chi2.ppf(0.025, 2 * k) / 2
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return {"k": int(k), "hours": hours, "per_hour": k / hours if hours else None,
            "lo": lo / hours if hours else None, "hi": hi / hours if hours else None}


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie): genuine (pos) vs attack (neg)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


# ------------------------------------------------------------------ turn windows
def base_yaw(frames):
    al = [f["yaw"] for f in frames if f["phase"] == "align" and f["yaw"] is not None]
    return float(np.median(al[-4:])) if al else 0.0


def turn_windows(rec, stat="rms"):
    """One dict per turn challenge of a session: per-frame yaw / landmark-parallax / flow progress."""
    tr = rec.get("_trace")
    if not tr or "frames" not in tr:
        return []
    frames = tr["frames"]
    by = base_yaw(frames)
    out, cur = [], None
    for f in frames:
        is_turn = f["phase"] == "challenge" and f["challenge"] in ("turn_left", "turn_right")
        key = (f["step"], f["challenge"]) if is_turn else None
        if cur is not None and key != cur["key"]:
            out.append(cur)
            cur = None
        if not is_turn:
            continue
        if cur is None:
            cur = {"key": key, "sign": 1.0 if f["challenge"] == "turn_left" else -1.0, "yaw": [], "par": [], "flow": [],
                   "t": []}
        s = cur["sign"]
        cur["t"].append(f["t"])
        cur["yaw"].append(np.nan if f["yaw"] is None else s * (f["yaw"] - by))
        cur["par"].append(np.nan if f["parallax"] is None else s * f["parallax"])
        fl = f.get("flow")
        cur["flow"].append(np.nan if fl is None or fl[FLOW_INDEX[stat]] is None else fl[FLOW_INDEX[stat]])
    if cur is not None:
        out.append(cur)
    for w in out:
        for k in ("yaw", "par", "flow", "t"):
            w[k] = np.asarray(w[k], float)
    return out


def held(mask, hold=HOLD):
    run = 0
    for m in mask:
        run = run + 1 if m else 0
        if run >= hold:
            return True
    return False


def window_outcome(w, flow_thr):
    yaw_ok = np.nan_to_num(w["yaw"], nan=-1e9) >= TURN
    par_ok = np.nan_to_num(w["par"], nan=-1e9) >= PAR_NEED
    flow_ok = np.nan_to_num(w["flow"], nan=-1e9) >= flow_thr if flow_thr else np.zeros_like(yaw_ok)
    return {"yaw": held(yaw_ok), "landmark": held(yaw_ok & par_ok), "flow": held(yaw_ok & flow_ok),
            "max_yaw": float(np.nanmax(w["yaw"])) if np.isfinite(w["yaw"]).any() else None,
            "max_par": float(np.nanmax(w["par"])) if np.isfinite(w["par"]).any() else None,
            "max_flow": float(np.nanmax(w["flow"])) if np.isfinite(w["flow"]).any() else None}


def matched_value(w, key, lo=0.8 * TURN, hi=1.6 * TURN):
    """Max of a cue over the frames whose apparent yaw change is in [lo, hi] (cue at matched apparent rotation)."""
    m = (w["yaw"] >= lo) & (w["yaw"] <= hi) & np.isfinite(w[key])
    return float(np.max(w[key][m])) if m.any() else None


# ------------------------------------------------------------------ freezing
def freeze(recs, stat="rms"):
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists - the threshold is already frozen; delete it only to restart the study.")
    if any(r.get("attack") for r in recs):
        print("note: attack trials already exist; they are NOT used for the threshold")
    cal = [r for r in recs if r.get("user") == "owner" and r["block"] in ("genuine_turns", "genuine_full")]
    vals = []
    for r in cal:
        for w in turn_windows(r, stat):
            idx = np.where(np.nan_to_num(w["yaw"], nan=-1e9) >= TURN)[0]
            if len(idx) and np.isfinite(w["flow"][idx[0]]):
                vals.append(float(w["flow"][idx[0]]))
    if len(vals) < 10:
        raise SystemExit(f"only {len(vals)} owner turns reached the turn threshold - record more calibration sessions")
    thr = 0.8 * float(np.median(vals))
    out = {"flow_statistic": stat, "flow_threshold": round(thr, 5),
           "rule": "0.8 x median planarity residual at the first frame where the yaw ratio reaches 18 deg "
                   "(owner's bona fide turns only)",
           "calibration": {"user": "owner", "sessions": len(cal), "turns": len(vals),
                           "median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                           "p90": float(np.percentile(vals, 90))},
           "frozen_at": datetime.now().isoformat(timespec="seconds")}
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------ analysis parts
def session_pad(rec):
    tr = rec.get("_trace") or {}
    by = defaultdict(list)
    lat = defaultdict(list)
    for t, name, p, ms in tr.get("pad", []):
        if p == p:  # not NaN
            by[name].append(p)
            lat[name].append(ms)
    return {k: float(np.mean(v)) for k, v in by.items()}, {k: v for k, v in lat.items()}


def pad_metrics(bona, attacks_by_species, score_fn, thr=0.5):
    """ISO/IEC 30107-3: APCER per species, BPCER; ACER = (max APCER + BPCER) / 2. score_fn -> P(attack) or None."""
    bs = [score_fn(r) for r in bona]
    bs = [s for s in bs if s is not None]
    out = {"bpcer": rate(sum(s >= thr for s in bs), len(bs)), "apcer": {}}
    all_att = []
    for sp, recs in attacks_by_species.items():
        ss = [score_fn(r) for r in recs]
        ss = [s for s in ss if s is not None]
        all_att += ss
        out["apcer"][sp] = rate(sum(s < thr for s in ss), len(ss))
    rates = [v["rate"] for v in out["apcer"].values() if v["rate"] is not None]
    out["apcer_max"] = max(rates) if rates else None
    out["acer"] = (out["apcer_max"] + out["bpcer"]["rate"]) / 2 if rates and out["bpcer"]["rate"] is not None else None
    out["auc"] = auc(all_att, bs)  # attack scores should be higher
    if bs and all_att:
        grid = np.unique(np.concatenate([bs, all_att]))
        apcer = np.array([np.mean(np.asarray(all_att) < g) for g in grid])
        bpcer = np.array([np.mean(np.asarray(bs) >= g) for g in grid])
        i = int(np.argmin(np.abs(apcer - bpcer)))
        out["eer"] = float((apcer[i] + bpcer[i]) / 2)
    return out


def guard_analysis(recs):
    expected = {"walk_left": "lock", "walk_right": "lock", "walk_behind": "lock", "walk_far": "lock",
                "early_return": "none", "occlude_hand": "none", "look_down": "none", "lights": "none",
                "bystander_pass": "none", "shoulder": "alert", "replacement": "intruder"}
    per_type = defaultdict(lambda: {"n": 0, "correct": 0, "latency_cue": [], "latency_last_seen": []})
    work_s, false_locks, false_alerts, sessions = 0.0, 0, 0, 0
    for r in recs:
        if r["block"] != "guard_session":
            continue
        sessions += 1
        tr = r.get("_trace") or {}
        fr = np.asarray([x for x in tr.get("frames", [])], dtype=object)
        times = np.array([x[0] for x in fr], float) if len(fr) else np.zeros(0)
        auth = np.array([x[2] for x in fr], int) if len(fr) else np.zeros(0, int)
        strangers = np.array([x[3] for x in fr], int) if len(fr) else np.zeros(0, int)
        acts = r["actions"]
        windows = []
        for e in r["events"]:
            if e.get("skipped") or "t_cue" not in e:
                continue
            a, b = e["t_cue"], e.get("t_end", e["end"]) + 15.0
            windows.append((a, b))
            kind = expected[e["event"]]
            inwin = [x for x in acts if a <= x["t"] <= b]
            locks = [x for x in inwin if x["kind"] == "lock"]
            alerts = [x for x in inwin if x["kind"] == "notify" and x["reason"] == "shoulder"]
            d = per_type[e["event"]]
            d["n"] += 1
            if kind == "lock":
                hit = [x for x in locks if x["reason"] == "absence"]
                d["correct"] += bool(hit)
                if hit:
                    d["latency_cue"].append(hit[0]["t"] - a)
                    seen = times[(times <= hit[0]["t"]) & (auth == 1)]
                    if len(seen):
                        d["latency_last_seen"].append(hit[0]["t"] - seen[-1])
            elif kind == "intruder":
                hit = [x for x in locks if x["reason"] == "intruder"]
                d["correct"] += bool(hit)
                if hit:
                    d["latency_cue"].append(hit[0]["t"] - a)
                    first = times[(times >= a) & (strangers > 0)]
                    if len(first):
                        d["latency_last_seen"].append(hit[0]["t"] - first[0])
            elif kind == "alert":
                d["correct"] += bool(alerts)
                if alerts:
                    d["latency_cue"].append(alerts[0]["t"] - a)
            else:
                d["correct"] += not locks and not alerts
        # work periods: everything outside the event windows (and the first 10 s)
        end = r["duration_s"]
        grid = np.arange(10.0, end, 1.0)
        outside = np.array([not any(a <= g <= b for a, b in windows) for g in grid])
        work_s += float(outside.sum())
        for x in acts:
            if any(a <= x["t"] <= b for a, b in windows) or x["t"] < 10.0:
                continue
            if x["kind"] == "lock":
                false_locks += 1
            elif x["kind"] == "notify" and x["reason"] in ("shoulder", "intruder"):
                false_alerts += 1
    out = {"sessions": sessions, "work_hours": work_s / 3600.0,
           "false_locks": poisson_rate(false_locks, work_s / 3600.0) if work_s else None,
           "false_alerts": poisson_rate(false_alerts, work_s / 3600.0) if work_s else None, "events": {}}
    for k, d in per_type.items():
        out["events"][k] = {"expected": expected[k], **rate(d["correct"], d["n"]),
                            "latency_cue_median": float(np.median(d["latency_cue"])) if d["latency_cue"] else None,
                            "latency_cue": [round(x, 1) for x in d["latency_cue"]],
                            "latency_last_seen": [round(x, 1) for x in d["latency_last_seen"]]}
    return out


def static_analysis(recs):
    out = {}
    for r in recs:
        if r["block"] != "static":
            continue
        fr = [f for f in r["_trace"]["frames"] if f["lm"] is not None and f["t"] > 1.0]
        if len(fr) < 10:
            continue
        lm = np.array([f["lm"] for f in fr], float)
        iod = np.median(np.linalg.norm(lm[:, 1] - lm[:, 0], axis=1))
        jitter = float(np.median(np.std(lm - lm.mean(axis=0), axis=0)))
        from faceguard.vision.geometry import parallax_yaw
        ref = np.median(lm[:5], axis=0)
        par = np.array([parallax_yaw(ref, x) for x in lm[5:]])
        flows = np.array([f["flow"][1] for f in fr if f.get("flow") and f["flow"][1] is not None])
        u = out.setdefault(r["user"], {"landmark_sd_px": [], "landmark_sd_iod": [], "parallax_sd": [], "flow_rms_p95": []})
        u["landmark_sd_px"].append(jitter)
        u["landmark_sd_iod"].append(jitter / iod)
        u["parallax_sd"].append(float(np.std(par)))
        if len(flows):
            u["flow_rms_p95"].append(float(np.percentile(flows, 95)))
    return out


def analyse(recs, stat):
    fz = json.loads(FROZEN.read_text()) if FROZEN.exists() else None
    thr = fz["flow_threshold"] if fz else None
    res = {"frozen": fz, "counts": defaultdict(int)}
    for r in recs:
        res["counts"][f"{r['block']}/{r.get('user')}"] += 1
    res["counts"] = dict(res["counts"])
    bona_full = [r for r in recs if r["block"] == "genuine_full"]
    bona_turn = [r for r in recs if r["block"] == "genuine_turns"]
    attacks = [r for r in recs if r.get("attack")]
    users = sorted({r["user"] for r in bona_full + bona_turn})

    # bona fide pass rates
    res["bona_fide"] = {
        "full": rate(sum(r["verified"] for r in bona_full), len(bona_full)),
        "full_by_user": {u: rate(sum(r["verified"] for r in bona_full if r["user"] == u),
                                 sum(r["user"] == u for r in bona_full)) for u in users},
        "turns_by_gate": {g: rate(sum(r["verified"] for r in bona_turn if r["gate"] == g),
                                  sum(r["gate"] == g for r in bona_turn)) for g in ("landmark", "flow")},
        "turns_heldout_by_gate": {g: rate(sum(r["verified"] for r in bona_turn if r["gate"] == g and r["user"] != "owner"),
                                          sum(r["gate"] == g and r["user"] != "owner" for r in bona_turn))
                                  for g in ("landmark", "flow")},
        "fps_median": float(np.median([r["fps"] for r in bona_full + bona_turn if r.get("fps")]))
        if bona_full + bona_turn else None,
        "users": users}

    # attack success: per instrument x protocol x gate
    species = sorted({r["pai"] for r in attacks})
    res["attacks"] = {}
    for sp in species:
        a = [r for r in attacks if r["pai"] == sp]
        res["attacks"][sp] = {
            "full": rate(sum(r["verified"] for r in a if r["block"] == "attack_full"),
                         sum(r["block"] == "attack_full" for r in a)),
            "full_live": rate(sum(r["live"] for r in a if r["block"] == "attack_full"),
                              sum(r["block"] == "attack_full" for r in a)),
            "turns": {g: rate(sum(r["verified"] for r in a if r["block"] == "attack_turns" and r["gate"] == g),
                              sum(r["block"] == "attack_turns" and r["gate"] == g for r in a)) for g in ("none", "landmark", "flow")},
            "operators": sorted({r["operator"] for r in a}), "media": sorted({r["media"] for r in a})}

    # paired per-challenge gate analysis and matched-yaw cue separation
    def windows(rs):
        return [(r, w) for r in rs for w in turn_windows(r, stat)]

    gw = windows(bona_full + bona_turn)
    res["paired"] = {}
    res["separation"] = {}
    g_par = [v for _, w in gw if (v := matched_value(w, "par")) is not None]
    g_flow = [v for _, w in gw if (v := matched_value(w, "flow")) is not None]
    gen_out = [window_outcome(w, thr) for _, w in gw]
    res["paired"]["bona_fide"] = {"turns": len(gen_out), "yaw_met": sum(o["yaw"] for o in gen_out),
                                  "landmark_pass": sum(o["landmark"] for o in gen_out if o["yaw"]),
                                  "flow_pass": sum(o["flow"] for o in gen_out if o["yaw"])}
    for sp in species + ["all_photo"]:
        rs = [r for r in attacks if (r["pai"] == sp if sp != "all_photo" else r["pai"] in ("phone_photo", "screen_photo"))]
        aw = windows(rs)
        outs = [window_outcome(w, thr) for _, w in aw]
        met = [o for o in outs if o["yaw"]]
        lm_block = sum(not o["landmark"] for o in met)
        fl_block = sum(not o["flow"] for o in met)
        res["paired"][sp] = {"turns": len(outs), "yaw_met": len(met),
                             "landmark_blocks": rate(lm_block, len(met)), "flow_blocks": rate(fl_block, len(met)),
                             "max_yaw": max([o["max_yaw"] for o in outs if o["max_yaw"] is not None], default=None),
                             "max_par": max([o["max_par"] for o in outs if o["max_par"] is not None], default=None)}
        if met:  # exact McNemar on the discordant pairs (landmark gate vs flow gate on the same attempts)
            b = sum(o["landmark"] and not o["flow"] for o in met)
            c = sum(o["flow"] and not o["landmark"] for o in met)
            res["paired"][sp]["mcnemar_p"] = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
        a_par = [v for _, w in aw if (v := matched_value(w, "par")) is not None]
        a_flow = [v for _, w in aw if (v := matched_value(w, "flow")) is not None]
        res["separation"][sp] = {"auc_landmark": auc(g_par, a_par), "auc_flow": auc(g_flow, a_flow),
                                 "n_genuine": [len(g_par), len(g_flow)], "n_attack": [len(a_par), len(a_flow)],
                                 "attack_par_median": float(np.median(a_par)) if a_par else None,
                                 "attack_flow_median": float(np.median(a_flow)) if a_flow else None}
    res["separation"]["genuine"] = {"par_median": float(np.median(g_par)) if g_par else None,
                                    "flow_median": float(np.median(g_flow)) if g_flow else None}

    # cue against apparent yaw (for the figure): binned medians
    def binned(pairs):
        bins = np.arange(0, 45, 5.0)
        out = []
        for lo in bins:
            sel = [v for y, v in pairs if lo <= y < lo + 5]
            out.append([float(lo + 2.5), len(sel)] + ([float(np.percentile(sel, q)) for q in (10, 50, 90)] if sel else [None] * 3))
        return out

    def frames_of(rs, key):
        return [(y, v) for _, w in windows(rs) for y, v in zip(w["yaw"], w[key]) if np.isfinite(y) and np.isfinite(v)]

    res["curves"] = {"genuine": {k: binned(frames_of(bona_full + bona_turn, k)) for k in ("par", "flow")}}
    for sp in species:
        rs = [r for r in attacks if r["pai"] == sp]
        res["curves"][sp] = {k: binned(frames_of(rs, k)) for k in ("par", "flow")}

    # PAD baselines and FaceGuard on identical sessions (full protocol: bona fide genuine_full, attacks attack_full)
    att_full = defaultdict(list)
    att_all = defaultdict(list)
    for r in attacks:
        att_all[r["pai"]].append(r)
        if r["block"] == "attack_full":
            att_full[r["pai"]].append(r)
    pad_cache = {id(r): session_pad(r) for r in recs}
    res["pad"] = {}
    for name in PAD_NAMES:
        fn = lambda r, n=name: pad_cache[id(r)][0].get(n)  # noqa: E731
        lat = [ms for r in recs for ms in pad_cache[id(r)][1].get(name, [])]
        res["pad"][name] = {"full_protocol_sessions": pad_metrics(bona_full, att_full, fn),
                            "all_sessions": pad_metrics(bona_full + bona_turn, att_all, fn),
                            "latency_ms_median": float(np.median(lat)) if lat else None}
    fg = lambda r: 0.0 if r["live"] else 1.0  # noqa: E731  FaceGuard's liveness decision as a score
    res["pad"]["faceguard"] = {"full_protocol_sessions": pad_metrics(bona_full, att_full, fg)}
    for name in PAD_NAMES:  # fusion: bona fide only if FaceGuard and the detector both accept
        fn = lambda r, n=name: (None if pad_cache[id(r)][0].get(n) is None  # noqa: E731
                                else max(fg(r), pad_cache[id(r)][0][n]))
        res["pad"][f"faceguard+{name}"] = {"full_protocol_sessions": pad_metrics(bona_full, att_full, fn)}

    res["static"] = static_analysis(recs)
    res["guard"] = guard_analysis(recs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--stat", default="rms", choices=["rms", "p75", "outlier_frac"])
    args = ap.parse_args()
    recs = load()
    if args.freeze:
        freeze(recs, args.stat)
        return
    res = analyse(recs, args.stat)
    save_json("physical", res)
    print(json.dumps({k: res[k] for k in ("counts", "bona_fide", "attacks", "paired")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
