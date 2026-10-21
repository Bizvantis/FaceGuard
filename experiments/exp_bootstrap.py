"""Identity-level (subject) bootstrap confidence intervals for the pair-based recognition results.

Pairs that share a person are not independent, so instead of treating the 1,000,000 LFW impostor pairs as
independent observations, each bootstrap replicate redraws the PEOPLE with replacement and weights every pair by the
product of its people's draw counts (genuine pairs: the person's count).

LFW: the operating-point pairs of exp_lfw.py (all genuine pairs with at most 50 images per person; 1,000,000 random
different-person pairs, seed 2026 drawn in the same order) at the three security levels and the adaptive-update gate.
MPIIFaceGaze: the cross-day pairwise protocol and the enrollment protocol of exp_mpii.py (SFace), bootstrapping the
15 participants; SFace embeddings of the 4,500 sampled frames are cached in results/mpii_sface.npz.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import exp_mpii
from common import RESULTS, lfw_embeddings, make_engine, save_json

from faceguard.vision.camera import load_image
from faceguard.vision.embedder import normalize

B, SEED = 1000, 2026
LEVELS = {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52, "update_gate": 0.57}
_ENG = None


def lfw_pairs_with_people(data):
    """Exactly the pairs of exp_lfw.operating_points, with the person index of each side."""
    rng = np.random.default_rng(SEED)
    E, names = data["e_tta"], data["names"]
    pid = {n: i for i, n in enumerate(dict.fromkeys(names))}
    person = np.array([pid[n] for n in names])
    by = defaultdict(list)
    for i, n in enumerate(names):
        by[n].append(i)
    gen, gen_p = [], []
    for n, idx in by.items():
        idx = idx[:50]
        if len(idx) > 1:
            s = E[idx] @ E[idx].T
            v = s[np.triu_indices(len(idx), 1)]
            gen.append(v)
            gen_p.append(np.full(len(v), pid[n]))
    a, b = rng.integers(len(E), size=2_000_000), rng.integers(len(E), size=2_000_000)
    diff = names[a] != names[b]
    a, b = a[diff][:1_000_000], b[diff][:1_000_000]
    imp = np.einsum("ij,ij->i", E[a], E[b])
    return np.concatenate(gen), np.concatenate(gen_p), imp, person[a], person[b], len(pid)


def bootstrap(gen, gp, imp, pa, pb, n_people, thresholds, b=B, seed=SEED + 7):
    rng = np.random.default_rng(seed)
    g_hit = {k: (gen < t) for k, t in thresholds.items()}
    i_hit = {k: (imp >= t) for k, t in thresholds.items()}
    reps = {k: [] for k in thresholds}
    for _ in range(b):
        c = np.bincount(rng.integers(0, n_people, n_people), minlength=n_people).astype(np.float64)
        gw, iw = c[gp], (c[pa] if pb is None else c[pa] * c[pb])
        gs, is_ = gw.sum(), iw.sum()
        for k in thresholds:
            reps[k].append((float(gw @ g_hit[k] / gs), float(iw @ i_hit[k] / is_)))
    out = {}
    for k, t in thresholds.items():
        r = np.array(reps[k])
        out[k] = {"threshold": t, "frr": float(np.mean(g_hit[k])), "far": float(np.mean(i_hit[k])),
                  "frr_ci": np.percentile(r[:, 0], [2.5, 97.5]).tolist(), "far_ci": np.percentile(r[:, 1], [2.5, 97.5]).tolist()}
    return out


def _init():
    global _ENG
    import cv2
    cv2.setNumThreads(1)
    _ENG = make_engine()


def _embed(path):
    img = load_image(path)
    faces = _ENG.detector.detect(img)
    if not faces:
        return None
    h, w = img.shape[:2]
    face = min(faces, key=lambda f: (f.center[0] - w / 2) ** 2 + (f.center[1] - h / 2) ** 2)
    _, emb, _ = _ENG.embed_face(img, face, tta=True)
    return normalize(np.asarray(emb, np.float32).reshape(-1))


def mpii():
    cache = RESULTS / "mpii_sface.npz"
    items = exp_mpii.sample()
    if cache.exists():
        z = np.load(cache)
        E, keep = z["E"], z["keep"]
    else:
        t0 = time.time()
        with ProcessPoolExecutor(4, initializer=_init) as pool:
            embs = list(pool.map(_embed, [str(f) for _, _, _, f in items], chunksize=16))
        print(f"embedded {len(items)} MPIIFaceGaze frames in {time.time() - t0:.0f}s")
        keep = np.array([i for i, e in enumerate(embs) if e is not None])
        E = np.stack([embs[i] for i in keep]).astype(np.float32)
        np.savez_compressed(cache, E=E, keep=keep)
    person = np.array([items[i][0] for i in keep])
    day = np.array([items[i][1] for i in keep])
    ndays = np.array([items[i][2] for i in keep])
    people = sorted(set(person))
    pnum = np.array([people.index(p) for p in person])
    # cross-day pairwise protocol
    iu = np.triu_indices(len(E), 1)
    s = (E @ E.T)[iu]
    same = person[iu[0]] == person[iu[1]]
    cross = day[iu[0]] != day[iu[1]]
    gsel, isel = same & cross, ~same
    pair = bootstrap(s[gsel], pnum[iu[0]][gsel], s[isel], pnum[iu[0]][isel], pnum[iu[1]][isel], len(people),
                     {"balanced": 0.45, "strict": 0.52, "relaxed": 0.38})
    # enrollment protocol: probe = later-day frame, scored against own gallery (genuine) and best other gallery
    gal, probes = {}, {}
    for p in people:
        idx = np.where(person == p)[0]
        order = idx[np.argsort(day[idx], kind="stable")]
        gal[p] = order[: exp_mpii.ENROLL]
        late = day[idx] >= ndays[idx] / 2.0
        probes[p] = np.setdiff1d(idx[late], gal[p])
    top = lambda q, p: float(np.sort(E[gal[p]] @ E[q])[::-1][: exp_mpii.K].mean())  # noqa: E731
    gen, gp, imp, ip = [], [], [], []
    for k, p in enumerate(people):
        for q in probes[p]:
            gen.append(top(q, p))
            gp.append(k)
            imp.append(max(top(q, o) for o in people if o != p))
            ip.append(k)
    gen, gp, imp, ip = map(np.array, (gen, gp, imp, ip))
    enroll = bootstrap(gen, gp, imp, ip, None, len(people), {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52})
    return {"people": len(people), "frames": int(len(E)), "pairwise": pair, "enrollment": enroll}


def main():
    data = lfw_embeddings()
    gen, gp, imp, pa, pb, n_people = lfw_pairs_with_people(data)
    print(f"LFW: {len(gen):,} genuine and {len(imp):,} impostor pairs over {n_people} people")
    out = {"bootstrap": B, "seed": SEED, "lfw": {"people": n_people, "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
                                                 "levels": bootstrap(gen, gp, imp, pa, pb, n_people, LEVELS)}}
    for k, v in out["lfw"]["levels"].items():
        print(f"  LFW {k:12s} FAR {100 * v['far']:.5f}% [{100 * v['far_ci'][0]:.5f}, {100 * v['far_ci'][1]:.5f}]  "
              f"FRR {100 * v['frr']:.2f}% [{100 * v['frr_ci'][0]:.2f}, {100 * v['frr_ci'][1]:.2f}]")
    out["mpii"] = mpii()
    for proto in ("pairwise", "enrollment"):
        for k, v in out["mpii"][proto].items():
            print(f"  MPII {proto:10s} {k:9s} FAR {100 * v['far']:.4f}% [{100 * v['far_ci'][0]:.4f}, {100 * v['far_ci'][1]:.4f}]  "
                  f"FRR {100 * v['frr']:.2f}% [{100 * v['frr_ci'][0]:.2f}, {100 * v['frr_ci'][1]:.2f}]")
    save_json("bootstrap", out)


if __name__ == "__main__":
    main()
