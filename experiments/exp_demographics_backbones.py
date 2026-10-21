"""Demographic differentials with identity-level statistics, balanced groups and three recognition backbones.

Extends exp_demographics.py (same person-level attribute labels, same 50-image cap) in three ways:

1. Backbones. SFace (the FaceGuard model) and the two InsightFace ArcFace models of exp_baselines.py, all with the
   same detections, five-landmark alignment and flip TTA. ArcFace embeddings of all 13,233 LFW images are cached in
   results/lfw_arcface.npz (computed once, ~5 min).
2. Operating point. Each backbone is evaluated at its own threshold that gives FMR = 1e-4 on 1,000,000 random
   different-person LFW pairs (seed 2026, as exp_lfw.py), so differences between groups are not an artefact of one
   model's score scale. SFace is also reported at the deployed balanced threshold 0.45.
3. Statistics. Pairs that share people are not independent, so uncertainty comes from an identity-level (subject)
   bootstrap: each replicate draws the same number of people per group with replacement (balanced: the size of the
   smallest group of that attribute) and weights every pair by the product of its two people's draw counts.
   Groups with fewer than 150 labeled people (Indian: 105) are excluded from the balanced comparison.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

import exp_baselines as eb
from common import RESULTS, lfw_embeddings, lfw_folder, save_json
from exp_demographics import load_attributes

SEED, B, MAX_PER_PERSON, IMP_POOL, MIN_PEOPLE, TARGET_FMR = 2026, 1000, 50, 1_000_000, 150, 1e-4
CACHE = RESULTS / "lfw_arcface.npz"


def arcface_embeddings(data):
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    _, folder = lfw_folder()
    paths = [folder / n / f"{n}_{int(i):04d}.jpg" for n, i in zip(data["names"], data["idx"])]
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=eb._init) as pool:
        embs = list(pool.map(eb._embed, [(p, data["mean_row"]) for p in paths], chunksize=32))
    print(f"embedded {len(paths)} images with ArcFace in {time.time() - t0:.0f}s")
    out = {k: np.stack([e[k] for e in embs]).astype(np.float32) for k in eb.MODELS}
    np.savez_compressed(CACHE, **out)
    return out


def global_threshold(E, names, rng, target=TARGET_FMR):
    """Threshold with the target FMR on 1M random different-person pairs (same draw as exp_lfw.py)."""
    a, b = rng.integers(len(E), size=2_000_000), rng.integers(len(E), size=2_000_000)
    diff = names[a] != names[b]
    a, b = a[diff][:1_000_000], b[diff][:1_000_000]
    imp = np.einsum("ij,ij->i", E[a], E[b])
    return float(np.quantile(imp, 1 - target))


def group_pairs(E, people, rng):
    """Genuine scores per person and a pool of within-group impostor pairs with their two people's indices."""
    gen = [np.array([float(E[a] @ E[b]) for a, b in combinations(idx, 2)], np.float32) for idx in people]
    imgs = np.concatenate([np.array(idx) for idx in people])
    owner = np.concatenate([np.full(len(idx), k) for k, idx in enumerate(people)])
    a = rng.integers(0, len(imgs), 3 * IMP_POOL)
    b = rng.integers(0, len(imgs), 3 * IMP_POOL)
    ok = owner[a] != owner[b]
    a, b = a[ok][:IMP_POOL], b[ok][:IMP_POOL]
    imp = np.einsum("ij,ij->i", E[imgs[a]], E[imgs[b]])
    return gen, imp, owner[a], owner[b]


def at_threshold(pre, t):
    """Per-person genuine counts below t, genuine pair counts, and impostor hits at t."""
    gen, imp, pa, pb = pre
    below = np.array([np.sum(g < t) for g in gen], np.float64)
    npairs = np.array([len(g) for g in gen], np.float64)
    return below, npairs, imp >= t, pa, pb


def rates(at, counts=None):
    """FNMR and FMR; with person draw counts, pairs are weighted (identity-level bootstrap)."""
    below, npairs, hit, pa, pb = at
    if counts is None:
        return float(below.sum() / npairs.sum()), float(np.mean(hit))
    w = counts[pa] * counts[pb]
    return float(counts @ below / max(counts @ npairs, 1)), float(np.sum(w * hit) / max(np.sum(w), 1))


def main():
    rng = np.random.default_rng(SEED)
    data = lfw_embeddings()
    names = np.array([str(n) for n in data["names"]])
    backbones = {"sface": data["e_tta"].astype(np.float32), **arcface_embeddings(data)}
    labels = load_attributes()
    by_person = defaultdict(list)
    for i, n in enumerate(names):
        by_person[n].append(i)
    out = {"seed": SEED, "bootstrap": B, "target_fmr": TARGET_FMR, "min_people": MIN_PEOPLE, "backbones": {}}
    for bb, E in backbones.items():
        t_glob = global_threshold(E, names, np.random.default_rng(SEED))
        thresholds = {"fmr1e-4": t_glob}
        if bb == "sface":
            thresholds["deployed_0.45"] = 0.45
        res = {"thresholds": thresholds, "attributes": {}}
        for attr in ("sex", "ethnicity"):
            groups = defaultdict(list)
            for person, idx in by_person.items():
                g = labels.get(person, {}).get(attr)
                if g:
                    groups[g].append(idx[:MAX_PER_PERSON])
            groups = {g: p for g, p in groups.items() if len(p) >= MIN_PEOPLE}
            n_bal = min(len(p) for p in groups.values())
            pre = {g: group_pairs(E, p, np.random.default_rng(SEED)) for g, p in sorted(groups.items())}
            at = {(g, tn): at_threshold(pre[g], t) for g in pre for tn, t in thresholds.items()}
            ares = {"balanced_people": n_bal, "groups": {}}
            boot = {g: {tn: [] for tn in thresholds} for g in pre}
            brng = np.random.default_rng(SEED + 1)
            for _ in range(B):
                for g, (gen, imp, pa, pb) in pre.items():
                    counts = np.bincount(brng.integers(0, len(gen), n_bal), minlength=len(gen)).astype(np.float64)
                    for tn in thresholds:
                        boot[g][tn].append(rates(at[(g, tn)], counts))
            for g, (gen, imp, pa, pb) in pre.items():
                r = {"people": len(gen), "genuine_pairs": int(sum(len(x) for x in gen)), "impostor_pairs": int(len(imp))}
                for tn, t in thresholds.items():
                    fnmr, fmr = rates(at[(g, tn)])
                    bs = np.array(boot[g][tn])
                    r[tn] = {"fnmr": fnmr, "fmr": fmr, "fnmr_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                             "fmr_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
                ares["groups"][g] = r
            # ratio of each group's FMR to the lowest-FMR group, with a bootstrap CI (same replicates)
            for tn in thresholds:
                fm = {g: np.array(boot[g][tn])[:, 1] for g in pre}
                ref = min(pre, key=lambda g: ares["groups"][g][tn]["fmr"])
                ares[f"ratios/{tn}"] = {"reference": ref, **{
                    g: {"ratio": ares["groups"][g][tn]["fmr"] / max(ares["groups"][ref][tn]["fmr"], 1e-12),
                        "ci": np.percentile(fm[g] / np.maximum(fm[ref], 1e-9), [2.5, 97.5]).tolist()}
                    for g in pre if g != ref}}
            res["attributes"][attr] = ares
            for g, r in ares["groups"].items():
                x = r["fmr1e-4"]
                print(f"{bb:12s} {attr:9s} {g:8s} people {r['people']:5d} | FMR {100 * x['fmr']:.4f}% "
                      f"[{100 * x['fmr_ci'][0]:.4f}, {100 * x['fmr_ci'][1]:.4f}]  FNMR {100 * x['fnmr']:.2f}% "
                      f"[{100 * x['fnmr_ci'][0]:.2f}, {100 * x['fnmr_ci'][1]:.2f}]")
            print(f"{bb:12s} {attr:9s} ratios: {ares['ratios/fmr1e-4']}")
        out["backbones"][bb] = res
    save_json("demographics_backbones", out)


if __name__ == "__main__":
    main()
