"""Physical vs. simulated parallax: what makes real flat presentations produce parallax, and how much.

1. Real data. Per-frame apparent yaw change and parallax yaw recorded by the production engine during the turn
   challenges of the real-world evaluation (results/realworld/traces): all valid phone-screen attacks of both rounds
   and the owner's genuine sessions with the corrected protocol.
2. Landmark noise. The same quantities for rendered flat prints and real heads (150 evaluation meshes, rotations of
   -60..60 deg) under isotropic landmark noise of 1-5 px; the noise level whose flat-print parallax best matches the
   real phone attacks (upper percentiles per apparent-yaw bin) is the "effective" noise of the real pipeline.
3. Lens distortion. The 15 calibrated laptop webcams of MPIIFaceGaze (intrinsics and distortion coefficients of
   each participant's camera): flat prints and heads rendered through each camera with cv2.projectPoints, without
   landmark noise, at the image centre and 30% of the half-width off-centre, isolate the parallax that distortion
   alone adds to a planar presentation.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import scipy.io as sio

from common import DATA, RESULTS, save_json
from exp_pose import ANGLES, DIST_MM, KEY, project, rot_yaw
from exp_realworld_analysis import INVALID_ROUND1

from faceguard.vision.geometry import parallax_ratio, pose_from_landmarks

K_RATIO = json.loads((RESULTS / "parallax.json").read_text())["k_ratio"]
K_PAR = json.loads((RESULTS / "parallax.json").read_text())["k_parallax"]
BINS = [(0, 5), (5, 10), (10, 15), (15, 20)]
RW = RESULTS / "realworld"


def real_frames():
    """(|apparent yaw change|, |parallax|) of every turn-challenge frame, for phone attacks and for the owner."""
    rows = []
    for fn, rnd in (("pilot_trials_protocol_v1.jsonl", 1), ("trials.jsonl", 2)):
        for line in (RW / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("trace") or r.get("id") in INVALID_ROUND1 or not r.get("clean_presentation", True):
                continue
            if r["block"].startswith("phone"):
                kind = "phone"
            elif r["block"] in ("owner_live", "owner_turns") and rnd == 2 and r["subject"] == "owner":
                kind = "owner"
            else:
                continue
            tr = json.loads((RW / r["trace"]).read_text())
            base = [f["yaw"] for f in tr if f["phase"] == "align" and f["yaw"] is not None]
            if not base:
                continue
            b = float(np.median(base))
            for f in tr:
                if f["phase"] == "challenge" and f["challenge"] in ("turn_left", "turn_right") \
                        and f["yaw"] is not None and f["parallax"] is not None:
                    rows.append((kind, r["id"], abs(f["yaw"] - b), abs(f["parallax"])))
    return rows


def deg(ratio, k):
    return np.degrees(np.arctan(np.asarray(ratio) / k))


def simulate(shapes, kind, noise, rng, render=project):
    """(|apparent yaw change|, |parallax yaw|) for every mesh and rotation, noise in px."""
    out = []
    for P in shapes:
        S = P if kind == "head" else np.concatenate([P[:, :2], np.zeros((len(P), 1))], axis=1)
        base = render(S)[KEY] + rng.normal(0, noise, (5, 2))
        g0 = pose_from_landmarks(base).yaw_ratio
        for a in ANGLES:
            cur = render(rot_yaw(S, float(a)))[KEY] + rng.normal(0, noise, (5, 2))
            dyaw = deg(pose_from_landmarks(cur).yaw_ratio, K_RATIO) - deg(g0, K_RATIO)
            out.append((float(abs(dyaw)), float(abs(deg(parallax_ratio(base, cur), K_PAR)))))
    return np.array(out)


def binned(arr):
    """Percentiles of |parallax| per |apparent yaw| bin."""
    res = {}
    for lo, hi in BINS:
        v = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi), 1]
        res[f"{lo}-{hi}"] = None if len(v) < 5 else {"n": int(len(v)), "p50": float(np.percentile(v, 50)),
                                                    "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                                                    "max": float(v.max())}
    return res


def mpii_cameras():
    cams = []
    for p in sorted((DATA / "mpiifacegaze" / "MPIIFaceGaze").glob("p[0-9][0-9]")):
        m = sio.loadmat(p / "Calibration" / "Camera.mat")
        cams.append((p.name, np.asarray(m["cameraMatrix"], float), np.asarray(m["distCoeffs"], float).ravel()))
    return cams


def distorted_render(Kc, dist, shift_x_mm):
    def render(S):
        pts = (S + np.array([shift_x_mm, 0.0, DIST_MM])).astype(np.float64)
        img, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, dist)
        return img.reshape(-1, 2)
    return render


def main():
    shapes_all = np.load(RESULTS / "face_shapes.npz")
    shapes = shapes_all["shapes"][shapes_all["eval_idx"]]
    rows = real_frames()
    real = {k: np.array([(a, p) for kind, _, a, p in rows if kind == k]) for k in ("phone", "owner")}
    out = {"k_ratio": K_RATIO, "k_parallax": K_PAR, "bins": [f"{lo}-{hi}" for lo, hi in BINS],
           "real": {k: {"frames": int(len(v)), "sessions": len({i for kind, i, _, _ in rows if kind == k}), "bins": binned(v)}
                    for k, v in real.items()}}
    # how |parallax| grows with |apparent yaw| (slope through the origin and correlation, frames with 5-20 deg)
    for k, v in real.items():
        sel = v[(v[:, 0] >= 5) & (v[:, 0] < 20)]
        out["real"][k]["slope_5_20"] = float(np.sum(sel[:, 0] * sel[:, 1]) / np.sum(sel[:, 0] ** 2))
        out["real"][k]["pearson_5_20"] = float(np.corrcoef(sel[:, 0], sel[:, 1])[0, 1])
    # 2. noise sweep
    out["noise"] = {}
    for noise in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        rng = np.random.default_rng(int(noise * 10) + 3)
        flat, head = simulate(shapes, "flat", noise, rng), simulate(shapes, "head", noise, rng)
        out["noise"][str(noise)] = {"flat": binned(flat), "head": binned(head)}
        for name, v in (("flat", flat), ("head", head)):
            sel = v[(v[:, 0] >= 5) & (v[:, 0] < 20)]
            if len(sel) > 10:
                out["noise"][str(noise)][f"{name}_slope_5_20"] = float(np.sum(sel[:, 0] * sel[:, 1]) / np.sum(sel[:, 0] ** 2))
    # effective noise: smallest squared log error between real phone and simulated flat p90 over bins with data
    def err(noise):
        e = []
        for b in out["bins"]:
            r, s = out["real"]["phone"]["bins"].get(b), out["noise"][noise]["flat"].get(b)
            if r and s and r["n"] >= 20:
                e.append((math.log(r["p90"] + 0.5) - math.log(s["p90"] + 0.5)) ** 2)
        return float(np.mean(e)) if e else float("inf")
    errs = {nz: err(nz) for nz in out["noise"]}
    out["noise_fit_error"] = errs
    out["effective_noise_px"] = float(min(errs, key=errs.get))
    # 3. lens distortion of 15 real laptop webcams (no landmark noise)
    cams, dist_res = mpii_cameras(), {}
    zero = np.random.default_rng(0)
    for name, Kc, dist in cams:
        w = 2 * Kc[0, 2]
        per = {}
        for label, frac in (("centre", 0.0), ("off_centre", 0.3)):
            shift = frac * (w / 2) * DIST_MM / Kc[0, 0]  # mm at the subject distance
            r = distorted_render(Kc, dist, shift)
            flat = simulate(shapes[:50], "flat", 0.0, zero, r)
            head = simulate(shapes[:50], "head", 0.0, zero, r)
            ideal = distorted_render(Kc, np.zeros(5), shift)
            flat_ideal = simulate(shapes[:50], "flat", 0.0, zero, ideal)
            sel = flat[:, 0] < 20
            per[label] = {"flat_parallax_max": float(flat[sel, 1].max()), "flat_parallax_p99": float(np.percentile(flat[sel, 1], 99)),
                          "flat_parallax_max_undistorted": float(flat_ideal[flat_ideal[:, 0] < 20, 1].max()),
                          "head_parallax_at_18_25": float(np.median(head[(head[:, 0] >= 18) & (head[:, 0] < 25), 1]))}
        dist_res[name] = {"fx": float(Kc[0, 0]), "width_px": float(w), "dist": dist.tolist(), **per}
        print(f"{name}: fx {Kc[0, 0]:.0f}  centre flat max {per['centre']['flat_parallax_max']:.2f} deg  "
              f"off-centre flat max {per['off_centre']['flat_parallax_max']:.2f} deg (undistorted "
              f"{per['off_centre']['flat_parallax_max_undistorted']:.2f})")
    out["mpii_cameras"] = dist_res
    vals = [v["off_centre"]["flat_parallax_max"] for v in dist_res.values()]
    out["distortion_summary"] = {"cameras": len(vals), "flat_max_off_centre_median": float(np.median(vals)),
                                 "flat_max_off_centre_max": float(max(vals)),
                                 "flat_max_centre_max": float(max(v["centre"]["flat_parallax_max"] for v in dist_res.values()))}
    for k in ("phone", "owner"):
        print(k, out["real"][k]["frames"], "frames", out["real"][k]["sessions"], "sessions", out["real"][k]["bins"])
    for nz in out["noise"]:
        print("noise", nz, "flat", {b: (v and round(v["p90"], 1)) for b, v in out["noise"][nz]["flat"].items()},
              "err", round(errs[nz], 3))
    print("effective noise", out["effective_noise_px"], "| distortion", out["distortion_summary"])
    save_json("parallax_real", out)


if __name__ == "__main__":
    main()
