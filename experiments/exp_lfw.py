"""Recognition experiments on Labeled Faces in the Wild (LFW), using the cached embeddings from common.py.

1. Verification, standard View-2 protocol (10 folds x 600 pairs), with ablations of flip test-time
   augmentation and landmark alignment.
2. Operating points (FAR / FRR) of the three security levels over all genuine pairs and 1M impostor pairs.
3. Closed-set identification vs. gallery size and matching strategy (158 people with >= 10 images).
4. Open-set identification: detection-and-identification rate (DIR) vs false-alarm rate against unseen people.
5. Machine-learning classifiers on top of the embeddings.
6. Enrollment design: pose-diverse vs frontal-only enrollment.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from common import key_index, lfw_embeddings, lfw_pairs, save_json, ten_fold_accuracy

from faceguard.recognition.analysis import auc_score, equal_error_rate, roc_curve, tar_at_far

REPS = 5
LEVELS = {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52}


def new_yaw(old_deg):  # re-express yaw in the final (3-D calibrated) scale: K 0.94 -> 0.690 (calibration half, exp_parallax.py)
    return np.degrees(np.arctan(np.tan(np.radians(old_deg)) * 0.94 / 0.690))


def topk_scores(probe, gallery, labels, n_cls, k):
    sims = probe @ gallery.T
    out = np.full((len(probe), n_cls), -1.0)
    for c in range(n_cls):
        s = sims[:, labels == c]
        kk = min(k, s.shape[1])
        out[:, c] = np.sort(s, axis=1)[:, -kk:].mean(axis=1) if kk else -1
    return out


def centroid_scores(probe, gallery, labels, n_cls):
    cen = np.stack([gallery[labels == c].mean(axis=0) for c in range(n_cls)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    return probe @ cen.T


def verification(data, key, pairs):
    out = {}
    same = np.array([s for _, _, s in pairs], dtype=bool)
    variants = {"full": "e_tta", "no_tta": "e_single", "no_alignment": "e_crop"}
    for name, field in variants.items():
        E = data[field]
        sims = np.array([float(E[key[a]] @ E[key[b]]) for a, b, _ in pairs])
        acc, std, thr = ten_fold_accuracy(sims, same)
        far, tar, _ = roc_curve(sims[same], sims[~same])
        eer, eer_thr = equal_error_rate(sims[same], sims[~same])
        keep = np.unique(np.r_[np.searchsorted(far, np.logspace(-4, 0, 200)), len(far) - 1].clip(0, len(far) - 1))
        out[name] = {"accuracy": acc, "std": std, "threshold": thr, "auc": auc_score(far, tar), "eer": eer,
                     "eer_threshold": eer_thr, "tar_at_far_1e-2": tar_at_far(sims[same], sims[~same], 1e-2)[0],
                     "tar_at_far_1e-3": tar_at_far(sims[same], sims[~same], 1e-3)[0],
                     "roc_far": far[keep].tolist(), "roc_tar": tar[keep].tolist(),
                     "genuine_mean": float(sims[same].mean()), "impostor_mean": float(sims[~same].mean())}
        print(f"  {name:13s} acc {100 * acc:.2f} +/- {100 * std:.2f}  AUC {out[name]['auc']:.4f}  EER {100 * eer:.2f}%")
    return out


def operating_points(data, rng):
    E, names = data["e_tta"], data["names"]
    by = defaultdict(list)
    for i, n in enumerate(names):
        by[n].append(i)
    gen = []
    for idx in by.values():
        idx = idx[:50]  # cap prolific people so they do not dominate
        if len(idx) > 1:
            s = E[idx] @ E[idx].T
            gen.extend(s[np.triu_indices(len(idx), 1)])
    gen = np.array(gen)
    a, b = rng.integers(len(E), size=2_000_000), rng.integers(len(E), size=2_000_000)
    diff = names[a] != names[b]
    a, b = a[diff][:1_000_000], b[diff][:1_000_000]
    imp = np.einsum("ij,ij->i", E[a], E[b])
    thr = np.round(np.arange(0.20, 0.71, 0.01), 2)
    curve = {"thresholds": thr.tolist(), "far": [float(np.mean(imp >= t)) for t in thr],
             "frr": [float(np.mean(gen < t)) for t in thr]}
    levels = {k: {"threshold": t, "far": float(np.mean(imp >= t)), "frr": float(np.mean(gen < t))} for k, t in LEVELS.items()}
    levels["unknown_zone_0.30"] = {"threshold": 0.30, "genuine_below": float(np.mean(gen < 0.30)), "impostor_below": float(np.mean(imp < 0.30))}
    levels["adaptive_gate_0.57"] = {"threshold": 0.57, "far": float(np.mean(imp >= 0.57)), "impostor_pairs": int(len(imp))}
    hist_edges = np.linspace(-0.3, 1.0, 66)
    hist = {"edges": hist_edges.tolist(), "genuine": (np.histogram(gen, hist_edges)[0] / len(gen)).tolist(),
            "impostor": (np.histogram(imp, hist_edges)[0] / len(imp)).tolist()}
    print(f"  genuine pairs {len(gen):,}  impostor pairs {len(imp):,}")
    for k, v in levels.items():
        print(f"  {k}: {v}")
    return {"n_genuine": int(len(gen)), "n_impostor": int(len(imp)), "curve": curve, "levels": levels, "hist": hist,
            "impostor_max": float(imp.max()), "impostor_p9999": float(np.quantile(imp, 0.9999))}


def subset(data, min_images):
    names = data["names"]
    counts = defaultdict(list)
    for i, n in enumerate(names):
        counts[n].append(i)
    people = sorted(n for n, idx in counts.items() if len(idx) >= min_images)
    return people, counts


def identification(data, rng):
    E = data["e_tta"]
    people, counts = subset(data, 10)
    res = defaultdict(list)
    for rep in range(REPS):
        for g in (1, 2, 3, 5, 8):
            gal, gl, pr, pl = [], [], [], []
            for c, n in enumerate(people):
                idx = rng.permutation(counts[n])
                gal += list(idx[:g]); gl += [c] * g
                pr += list(idx[g:]); pl += [c] * (len(idx) - g)
            G, P, gl, pl = E[gal], E[pr], np.array(gl), np.array(pl)
            for strat, fn in (("nn", lambda: topk_scores(P, G, gl, len(people), 1)),
                              ("top3", lambda: topk_scores(P, G, gl, len(people), 3)),
                              ("top5", lambda: topk_scores(P, G, gl, len(people), 5)),
                              ("centroid", lambda: centroid_scores(P, G, gl, len(people)))):
                if g == 1 and strat != "nn":
                    continue
                res[f"g{g}/{strat}"].append(float(np.mean(np.argmax(fn(), axis=1) == pl)))
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    out["people"], out["images"] = len(people), int(sum(len(counts[n]) for n in people))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  rank-1 {k:12s} {100 * v['mean']:.2f} +/- {100 * v['std']:.2f}")
    return out


def open_set(data, rng):
    E, names = data["e_tta"], data["names"]
    people, counts = subset(data, 10)
    thr = np.linspace(0.0, 1.0, 1001)
    agg = defaultdict(list)
    curves = {}
    for rep in range(REPS):
        enrolled = [people[i] for i in rng.permutation(len(people))[: len(people) // 2]]
        eset = set(enrolled)
        gal, gl, kp, kl = [], [], [], []
        for c, n in enumerate(enrolled):
            idx = rng.permutation(counts[n])
            gal += list(idx[:5]); gl += [c] * 5
            kp += list(idx[5:]); kl += [c] * (len(idx) - 5)
        unknown = np.array([i for i, n in enumerate(names) if n not in eset])
        G, gl, kl = E[gal], np.array(gl), np.array(kl)
        for strat, k in (("nn", 1), ("top3", 3), ("centroid", None)):
            sk = topk_scores(E[kp], G, gl, len(enrolled), k) if k else centroid_scores(E[kp], G, gl, len(enrolled))
            su = topk_scores(E[unknown], G, gl, len(enrolled), k) if k else centroid_scores(E[unknown], G, gl, len(enrolled))
            best_k, pred = sk.max(axis=1), sk.argmax(axis=1)
            correct = pred == kl
            best_u = su.max(axis=1)
            dir_ = np.array([np.mean(correct & (best_k >= t)) for t in thr])
            far = np.array([np.mean(best_u >= t) for t in thr])
            for target in (1e-2, 1e-3):
                ok = np.where(far <= target)[0]
                agg[f"{strat}/dir_at_far_{target:g}"].append(float(dir_[ok[0]]) if len(ok) else 0.0)
            for lvl, t in LEVELS.items():
                i = int(round(t * 1000))
                agg[f"{strat}/{lvl}/dir"].append(float(dir_[i]))
                agg[f"{strat}/{lvl}/far"].append(float(far[i]))
            agg[f"{strat}/closed_set_rank1"].append(float(np.mean(correct)))
            if rep == 0:
                keep = np.linspace(0, len(thr) - 1, 201).astype(int)
                curves[strat] = {"far": far[keep].tolist(), "dir": dir_[keep].tolist()}
        agg["unknown_probes"].append(len(unknown))
        agg["known_probes"].append(len(kp))
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in agg.items()}
    out["curves"] = curves
    out["enrolled_people"] = len(people) // 2
    for k, v in sorted(out.items()):
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k:32s} {v['mean']:.4f} +/- {v['std']:.4f}")
    return out


def classifiers(data, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    E = data["e_tta"]
    people, counts = subset(data, 10)
    res = defaultdict(list)
    for rep in range(REPS):
        gal, gl, pr, pl = [], [], [], []
        for c, n in enumerate(people):
            idx = rng.permutation(counts[n])
            gal += list(idx[:5]); gl += [c] * 5
            pr += list(idx[5:]); pl += [c] * (len(idx) - 5)
        G, P, gl, pl = E[gal], E[pr], np.array(gl), np.array(pl)
        res["Top-3 cosine (FaceGuard)"].append(np.mean(topk_scores(P, G, gl, len(people), 3).argmax(1) == pl))
        res["1-NN cosine"].append(np.mean(topk_scores(P, G, gl, len(people), 1).argmax(1) == pl))
        res["Nearest centroid"].append(np.mean(centroid_scores(P, G, gl, len(people)).argmax(1) == pl))
        for name, model in (("Linear SVM", LinearSVC(C=1.0)), ("k-NN (k=5, cosine)", KNeighborsClassifier(5, metric="cosine")),
                            ("Logistic regression", LogisticRegression(C=10.0, max_iter=3000))):
            res[name].append(float(np.mean(model.fit(G, gl).predict(P) == pl)))
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    for k, v in out.items():
        print(f"  {k:26s} {100 * v['mean']:.2f} +/- {100 * v['std']:.2f}")
    return out


def pose_enrollment(data, rng):
    E = data["e_tta"]
    yaw = new_yaw(data["yaw"])
    people, counts = subset(data, 20)
    res = defaultdict(list)
    for rep in range(REPS):
        for strategy in ("frontal", "diverse", "random"):
            gal, gl, pr, pl, pyaw = [], [], [], [], []
            for c, n in enumerate(people):
                idx = np.array(counts[n])
                order = idx[np.argsort(yaw[idx])]
                if strategy == "frontal":
                    g = idx[np.argsort(np.abs(yaw[idx]))][:5]
                elif strategy == "diverse":
                    q = np.linspace(0, len(order) - 1, 7)[1:-1].round().astype(int)  # yaw quantiles
                    g = order[q]
                else:
                    g = rng.choice(idx, 5, replace=False)
                rest = np.setdiff1d(idx, g)
                gal += list(g); gl += [c] * 5
                pr += list(rest); pl += [c] * len(rest); pyaw += list(yaw[rest])
            G, P, gl, pl, pyaw = E[gal], E[pr], np.array(gl), np.array(pl), np.abs(np.array(pyaw))
            sc = topk_scores(P, G, gl, len(people), 3)
            own = sc[np.arange(len(pl)), pl]
            ok = sc.argmax(1) == pl
            side = pyaw > 15
            res[f"{strategy}/rank1"].append(float(ok.mean()))
            res[f"{strategy}/rank1_nonfrontal"].append(float(ok[side].mean()))
            res[f"{strategy}/own_score_nonfrontal"].append(float(own[side].mean()))
            res[f"{strategy}/own_score_frontal"].append(float(own[~side].mean()))
            res["nonfrontal_fraction"].append(float(side.mean()))
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    out["people"] = len(people)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:32s} {v['mean']:.4f} +/- {v['std']:.4f}")
    return out


def dataset_stats(data):
    yaw = new_yaw(data["yaw"])
    names = data["names"]
    _, counts = subset(data, 1)
    return {"images": int(len(names)), "people": int(len(counts)),
            "people_ge2": int(sum(len(v) >= 2 for v in counts.values())), "people_ge10": int(sum(len(v) >= 10 for v in counts.values())),
            "people_ge20": int(sum(len(v) >= 20 for v in counts.values())),
            "detected": int((~data["fallback"]).sum()), "multi_face_images": int((data["n_faces"] > 1).sum()),
            "face_width_px_median": float(np.median(data["face_w"])), "abs_yaw_median": float(np.median(np.abs(yaw))),
            "abs_yaw_p90": float(np.percentile(np.abs(yaw), 90)), "sharpness_median": float(np.median(data["sharp"]))}


def main():
    rng = np.random.default_rng(2026)
    data = lfw_embeddings()
    key = key_index(data)
    pairs, _, _ = lfw_pairs()
    out = {"dataset": dataset_stats(data)}
    print(out["dataset"])
    print("verification (View 2)...")
    out["verification"] = verification(data, key, pairs)
    print("operating points...")
    out["operating_points"] = operating_points(data, rng)
    print("identification...")
    out["identification"] = identification(data, rng)
    print("open set...")
    out["open_set"] = open_set(data, rng)
    print("classifiers...")
    out["classifiers"] = classifiers(data, rng)
    print("pose-diverse enrollment...")
    out["pose_enrollment"] = pose_enrollment(data, rng)
    save_json("lfw", out)


if __name__ == "__main__":
    main()
