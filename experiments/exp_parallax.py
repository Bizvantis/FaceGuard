"""Homography-parallax liveness cue vs. the 5-landmark yaw ratio, on real 3-D face geometry.

For each of 300 near-frontal LFW faces (MediaPipe 478-point mesh, scaled to a 63 mm inter-ocular distance)
we render, through a 640x480 / 70 deg webcam model at 50 cm:
  * a real head turning by theta,
  * a flat print rotated by theta,
  * prints bent into a cylinder (radius R) and rotated by theta (warped-photo attack),
add 1 px landmark noise, and compute
  * the yaw-ratio estimate (nose offset from the eye midpoint), and
  * the parallax estimate: nose position minus its prediction from the homography that maps the
    baseline eye/mouth corners onto the current ones (exactly 0 for anything planar).
"""

from __future__ import annotations

import math
import random

import numpy as np

from common import RESULTS, lfw_index, save_json
from exp_pose import ANGLES, IOD_MM, KEY, landmarker, mesh3d, project, rot_yaw

from faceguard.vision.camera import load_image
from faceguard.vision.geometry import parallax_ratio, pose_from_landmarks

NOISE_PX = 1.0
BENDS_MM = (60.0, 40.0, 30.0)


def canonicalize(shapes: list[np.ndarray], iterations: int = 3) -> list[np.ndarray]:
    """Generalised Procrustes (rotation only): align every mesh to the mean face so all start frontal."""
    def rot(P, Q):  # rotation R minimising |P R^T - Q| (Kabsch)
        u, _, vt = np.linalg.svd(P.T @ Q)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        return vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    out = [s - s.mean(axis=0) for s in shapes]
    for _ in range(iterations):
        mean = np.mean(out, axis=0)
        out = [s @ rot(s, mean).T for s in out]
    return out


def bend(P, radius):
    Q = P.copy()
    Q[:, 2] = Q[:, 0] ** 2 / (2.0 * radius)  # cylinder bent towards the camera: the middle sticks out
    Q[:, 2] -= Q[:, 2].mean()
    return Q


SPLIT_SEED = 2026  # calibration / evaluation split of the 300 meshes (disjoint halves)


def build_shapes():
    rng = random.Random(0)
    mp, lm = landmarker()
    shapes: list[np.ndarray] = []
    for _, _, path in rng.sample(lfw_index(), 1500):
        if len(shapes) >= 300:
            break
        pts, ang = mesh3d(mp, lm, load_image(path))
        if pts is None or abs(ang[0]) > 10 or abs(ang[1]) > 15:  # near-frontal: |yaw| <= 10 deg, |pitch| <= 15 deg
            continue
        iod = np.linalg.norm(pts[KEY[1], :2] - pts[KEY[0], :2])
        shapes.append((pts - pts.mean(axis=0)) * (IOD_MM / iod))
    return np.stack(canonicalize(shapes))  # remove the head pose each photo was taken with


def measure(shapes, kinds, nrng):
    ratio = {k: [] for k in kinds}
    par = {k: [] for k in kinds}
    for P in shapes:
        for name, make in kinds.items():
            S = make(P)
            base = project(S)[KEY] + nrng.normal(0, NOISE_PX, (5, 2))
            r_row, p_row = [], []
            for a in ANGLES:
                cur = project(rot_yaw(S, a))[KEY] + nrng.normal(0, NOISE_PX, (5, 2))
                r_row.append(pose_from_landmarks(cur).yaw_ratio - pose_from_landmarks(base).yaw_ratio)
                p_row.append(parallax_ratio(base, cur))
            ratio[name].append(r_row)
            par[name].append(p_row)
    return {k: np.array(v) for k, v in ratio.items()}, {k: np.array(v) for k, v in par.items()}


def fit_k(values, small, tan):
    """Least squares for value ~= k * tan(theta) over the head rotations within +-30 deg."""
    return float(np.sum(values[:, small] * tan[None, :]) / (len(values) * np.sum(tan * tan)))


def main():
    path = RESULTS / "face_shapes.npz"
    if path.exists():
        shapes = np.load(path)["shapes"]
    else:
        shapes = build_shapes()
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(shapes))
    cal_idx, eval_idx = np.sort(perm[: len(shapes) // 2]), np.sort(perm[len(shapes) // 2:])
    # reused by exp_liveness.py, which only renders the evaluation half
    np.savez_compressed(path, shapes=shapes, cal_idx=cal_idx, eval_idx=eval_idx)
    kinds = {"head": lambda P: P, "flat": lambda P: np.concatenate([P[:, :2], np.zeros((len(P), 1))], axis=1)}
    for r in BENDS_MM:
        kinds[f"bent_R{int(r)}"] = lambda P, r=r: bend(np.concatenate([P[:, :2], np.zeros((len(P), 1))], axis=1), r)
    A = ANGLES.astype(float)
    small = np.abs(A) <= 30
    tan = np.tan(np.radians(A[small]))
    # calibration half: fit K (yaw ratio) and Kp (parallax) for real heads only
    cal_ratio, cal_par = measure(shapes[cal_idx], {"head": kinds["head"]}, np.random.default_rng(1))
    # evaluation half: every presentation, reported with the calibration-half constants
    ratio, par = measure(shapes[eval_idx], kinds, np.random.default_rng(0))
    nose_depth = [float(P[KEY[:2], 2].mean() - P[KEY[2], 2]) for P in shapes]  # mm in front of the eyes
    out = {"faces": int(len(shapes)), "faces_cal": int(len(cal_idx)), "faces_eval": int(len(eval_idx)),
           "split_seed": SPLIT_SEED, "angles": ANGLES.tolist(), "noise_px": NOISE_PX,
           "nose_depth_mm_mean": float(np.mean(nose_depth)), "nose_depth_mm_std": float(np.std(nose_depth))}
    out["k_ratio"] = fit_k(cal_ratio["head"], small, tan)
    out["k_parallax"] = fit_k(cal_par["head"], small, tan)
    # the same fit on the evaluation half shows how stable the constants are
    out["k_ratio_eval_fit"] = fit_k(ratio["head"], small, tan)
    out["k_parallax_eval_fit"] = fit_k(par["head"], small, tan)
    for name in kinds:
        R, Pp = ratio[name], par[name]
        deg_r = np.degrees(np.arctan(R / out["k_ratio"]))
        deg_p = np.degrees(np.arctan(Pp / out["k_parallax"]))
        out[name] = {
            "ratio_deg_mean": deg_r.mean(0).tolist(), "ratio_deg_std": deg_r.std(0).tolist(),
            "parallax_deg_mean": deg_p.mean(0).tolist(), "parallax_deg_std": deg_p.std(0).tolist(),
            "ratio_max_abs_deg": float(np.abs(deg_r).max()), "parallax_max_abs_deg": float(np.abs(deg_p).max()),
            "parallax_p99_abs_deg": float(np.percentile(np.abs(deg_p), 99)),
        }
        print(f"{name:10s} ratio max {out[name]['ratio_max_abs_deg']:6.1f} deg | parallax max {out[name]['parallax_max_abs_deg']:6.1f}"
              f"  p99 {out[name]['parallax_p99_abs_deg']:5.1f}  (at 25 deg: ratio {np.mean(deg_r[:, list(ANGLES).index(25)]):5.1f}, "
              f"parallax {np.mean(deg_p[:, list(ANGLES).index(25)]):5.1f})")
    print(f"k_ratio {out['k_ratio']:.3f} (eval-half fit {out['k_ratio_eval_fit']:.3f})  k_parallax {out['k_parallax']:.3f}"
          f" (eval-half fit {out['k_parallax_eval_fit']:.3f})  nose depth {out['nose_depth_mm_mean']:.1f} mm")
    save_json("parallax", out)


if __name__ == "__main__":
    main()
