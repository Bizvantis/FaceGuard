"""Head-pose experiments.

A. Planar-attack invariance: real 3-D face geometry (MediaPipe 478-point mesh from LFW photos) is either
   rotated as a head, or flattened onto a card that is rotated the same way (print attack). Both are
   projected through a pinhole webcam model and the 5-landmark yaw estimator is applied.
B. Agreement of the 5-landmark yaw/pitch with MediaPipe's 3-D head pose on real LFW images.
C. Measurement noise of the 5-landmark yaw under sensor noise (used by the liveness simulation).
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from common import lfw_index, make_engine, save_json

from faceguard import models
from faceguard.vision.camera import load_image
from faceguard.vision.geometry import pose_from_landmarks
from faceguard.vision.mesh import euler_from_matrix

KEY = [468, 473, 1, 61, 291]  # right-eye iris, left-eye iris, nose tip, right / left mouth corner (subject's view)
IOD_MM, DIST_MM = 63.0, 500.0
F_PX, CX, CY = 457.0, 320.0, 240.0  # 640x480 webcam with ~70 deg horizontal field of view
ANGLES = np.arange(-60, 61, 5)


def landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=models.MODELS["landmarker"].path.read_bytes()),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_facial_transformation_matrixes=True)
    return mp, vision.FaceLandmarker.create_from_options(opts)


def mesh3d(mp, lm, img):
    h, w = img.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return None, None
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in r.face_landmarks[0]], np.float64)
    return pts, euler_from_matrix(r.facial_transformation_matrixes[0])


def project(P):
    return np.stack([F_PX * P[:, 0] / (DIST_MM + P[:, 2]) + CX, F_PX * P[:, 1] / (DIST_MM + P[:, 2]) + CY], axis=1)


def rot_yaw(P, deg):
    a = math.radians(deg)  # positive = subject turns to their left (nose moves to the image's right)
    x, z = P[:, 0], P[:, 2]
    return np.stack([x * math.cos(a) - z * math.sin(a), P[:, 1], x * math.sin(a) + z * math.cos(a)], axis=1)


def rot_pitch(P, deg):
    a = math.radians(deg)  # positive = looking up (nose moves towards the eyes in the image)
    y, z = P[:, 1], P[:, 2]
    return np.stack([P[:, 0], y * math.cos(a) + z * math.sin(a), -y * math.sin(a) + z * math.cos(a)], axis=1)


def main():
    rng = random.Random(0)
    index = lfw_index()
    mp, lm = landmarker()
    engine = make_engine()

    # ---------------- A: planar vs 3-D
    yaw3d, yawflat, pitch3d, pitchflat = [], [], [], []
    candidates = rng.sample(index, 1500)
    for _, _, path in candidates:
        if len(yaw3d) >= 300:
            break
        img = load_image(path)
        pts, ang = mesh3d(mp, lm, img)
        if pts is None or abs(ang[0]) > 10 or abs(ang[1]) > 15:
            continue
        iod = np.linalg.norm(pts[KEY[1], :2] - pts[KEY[0], :2])
        P = (pts - pts.mean(axis=0)) * (IOD_MM / iod)
        flat = P.copy()
        flat[:, 2] = 0.0
        for store, rot, src in ((yaw3d, rot_yaw, P), (yawflat, rot_yaw, flat), (pitch3d, rot_pitch, P), (pitchflat, rot_pitch, flat)):
            row = []
            for a in ANGLES:
                q = project(rot(src, a))[KEY]
                pose = pose_from_landmarks(q)
                row.append(pose.yaw if rot is rot_yaw else pose.pitch)
            store.append(row)
    Y3, YF, P3, PF = (np.array(v) for v in (yaw3d, yawflat, pitch3d, pitchflat))
    zero = int(np.where(ANGLES == 0)[0][0])
    dYF = YF - YF[:, [zero]]  # relative to the pose at the start (liveness uses a per-session baseline)
    dY3 = Y3 - Y3[:, [zero]]
    nz = ANGLES != 0
    slopes_flat = np.array([np.polyfit(ANGLES, row, 1)[0] for row in dYF])
    slopes_3d = np.array([np.polyfit(ANGLES[np.abs(ANGLES) <= 30], row[np.abs(ANGLES) <= 30], 1)[0] for row in dY3])
    out = {
        "faces": int(len(Y3)),
        "angles": ANGLES.tolist(),
        "yaw_3d_mean": dY3.mean(0).tolist(), "yaw_3d_std": dY3.std(0).tolist(),
        "yaw_flat_mean": dYF.mean(0).tolist(), "yaw_flat_std": dYF.std(0).tolist(),
        "pitch_3d_mean": (P3 - P3[:, [zero]]).mean(0).tolist(), "pitch_flat_mean": (PF - PF[:, [zero]]).mean(0).tolist(),
        "flat_max_abs_delta_yaw": float(np.abs(dYF).max()),
        "flat_p99_abs_delta_yaw": float(np.percentile(np.abs(dYF), 99)),
        "flat_max_abs_delta_yaw_within_45": float(np.abs(dYF[:, np.abs(ANGLES) <= 45]).max()),
        "planar_gain_abs_max": float(np.abs(slopes_flat).max()),
        "planar_gain_p99": float(np.percentile(np.abs(slopes_flat), 99)),
        "head_gain_mean_within_30": float(slopes_3d.mean()), "head_gain_std_within_30": float(slopes_3d.std()),
        "head_angle_to_reach_18deg_mean": float(np.mean([ANGLES[np.argmax(row >= 18)] if (row >= 18).any() else np.nan for row in dY3])),
        "flat_fraction_ever_reaching_18deg": float(np.mean((np.abs(dYF) >= 18).any(axis=1))),
    }
    print(f"A: {out['faces']} faces; flat max |dyaw| {out['flat_max_abs_delta_yaw']:.2f} deg; head gain {out['head_gain_mean_within_30']:.3f}")

    # ---------------- B: agreement with MediaPipe 3-D pose on real photos
    ys, ym, ps, pm = [], [], [], []
    for _, _, path in rng.sample(index, 3000):
        img = load_image(path)
        faces = engine.detector.detect(img)
        if not faces:
            continue
        h, w = img.shape[:2]
        face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
        _, ang = mesh3d(mp, lm, img)
        if ang is None:
            continue
        ys.append(face.pose.yaw), ym.append(ang[0]), ps.append(face.pose.pitch), pm.append(ang[1])
    ys, ym, ps, pm = map(np.array, (ys, ym, ps, pm))
    out["agreement"] = {
        "n": int(len(ys)),
        "yaw_pearson_r": float(np.corrcoef(ys, ym)[0, 1]), "yaw_slope": float(np.polyfit(ym, ys, 1)[0]),
        "yaw_mae_deg": float(np.mean(np.abs(ys - ym))),
        "pitch_pearson_r": float(np.corrcoef(ps, pm)[0, 1]),
        "yaw_range_mediapipe": [float(ym.min()), float(ym.max())],
        "scatter_sample": {"yaw5": ys[:800].tolist(), "yawmp": ym[:800].tolist()},
    }
    print(f"B: n={len(ys)} yaw r={out['agreement']['yaw_pearson_r']:.3f} slope={out['agreement']['yaw_slope']:.3f} "
          f"MAE={out['agreement']['yaw_mae_deg']:.2f}; pitch r={out['agreement']['pitch_pearson_r']:.3f}")

    # ---------------- C: estimator noise under sensor noise
    nrng = np.random.default_rng(1)
    stds = []
    for _, _, path in rng.sample(index, 120):
        img = load_image(path).astype(np.int16)
        vals = []
        for _ in range(20):
            noisy = np.clip(img + nrng.normal(0, 4.0, img.shape), 0, 255).astype(np.uint8)
            f = engine.detector.detect(noisy)
            if f:
                h, w = noisy.shape[:2]
                vals.append(min(f, key=lambda q: (q.center[0] - w / 2) ** 2 + (q.center[1] - h / 2) ** 2).pose.yaw)
        if len(vals) >= 10:
            stds.append(np.std(vals))
    out["yaw_noise"] = {"images": len(stds), "sensor_sigma": 4.0, "mean_std_deg": float(np.mean(stds)),
                        "p95_std_deg": float(np.percentile(stds, 95))}
    print(f"C: yaw jitter mean {out['yaw_noise']['mean_std_deg']:.2f} deg, p95 {out['yaw_noise']['p95_std_deg']:.2f}")
    save_json("pose", out)


if __name__ == "__main__":
    main()
