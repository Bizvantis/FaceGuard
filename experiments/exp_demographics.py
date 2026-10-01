"""Demographic differentials of the SFace pipeline on LFW.

Labels: the LFW attribute file of Kumar et al. (classifier scores for 73 attributes, data/lfw_attributes.txt).
They are automatic estimates, so we label each PERSON by averaging the scores over all of their images:
sex = sign of "Male"; perceived ethnicity = argmax of {Asian, White, Black, Indian}; age = argmax of
{Youth, Middle Aged, Senior}. A person is labeled only when the average is decisive (|Male| >= 0.5; ethnicity
margin >= 0.5 over the runner-up), otherwise excluded from that attribute.

Per group (cached flip-TTA embeddings, results/lfw_embeddings.npz; seed 2026):
  FNMR = same-person pairs with similarity < tau (at most 50 images per person, as in exp_lfw.py);
  FMR  = different-person pairs WITHIN the group (same-demographic impostors are the hard case) >= tau,
         up to 1,000,000 random pairs per group.
Reported at the three security levels with 95% Wilson intervals, plus the max/min ratio across groups.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from common import DATA, lfw_embeddings, save_json

THRESHOLDS = {"relaxed": 0.38, "balanced": 0.45, "strict": 0.52}
SEED, MAX_IMP, MAX_PER_PERSON = 2026, 1_000_000, 50
ETH = ("Asian", "White", "Black", "Indian")
AGE = ("Youth", "Middle Aged", "Senior")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load_attributes():
    lines = (DATA / "lfw_attributes.txt").read_text(encoding="utf-8").splitlines()
    header = lines[1].lstrip("#").strip().split("\t")
    per = defaultdict(list)
    for ln in lines[2:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        per[row["person"].replace(" ", "_")].append({k: float(row[k]) for k in ("Male",) + ETH + AGE})
    labels = {}
    for person, rows in per.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        lab = {}
        if abs(m["Male"]) >= 0.5:
            lab["sex"] = "male" if m["Male"] > 0 else "female"
        eth = sorted(ETH, key=lambda k: -m[k])
        if m[eth[0]] - m[eth[1]] >= 0.5:
            lab["ethnicity"] = eth[0].lower()
        age = sorted(AGE, key=lambda k: -m[k])
        if m[age[0]] - m[age[1]] >= 0.5:
            lab["age"] = age[0].lower()
        labels[person] = lab
    return labels


def main():
    rng = np.random.default_rng(SEED)
    data = lfw_embeddings()
    E, names = data["e_tta"], np.array([str(n) for n in data["names"]])
    labels = load_attributes()
    by_person = defaultdict(list)
    for i, n in enumerate(names):
        by_person[n].append(i)
    out = {"seed": SEED, "labeled_people": len(labels), "attributes": {}}
    for attr in ("sex", "ethnicity", "age"):
        groups = defaultdict(list)
        for person, idx in by_person.items():
            g = labels.get(person, {}).get(attr)
            if g:
                groups[g].append(idx[:MAX_PER_PERSON])
        res = {}
        for g, people in sorted(groups.items()):
            gen = np.array([float(E[a] @ E[b]) for idx in people for a, b in combinations(idx, 2)])
            imgs = np.concatenate([np.array(idx) for idx in people])
            owner = np.concatenate([np.full(len(idx), k) for k, idx in enumerate(people)])
            n_imp = min(MAX_IMP, len(imgs) * (len(imgs) - 1) // 2)
            a = rng.integers(0, len(imgs), 3 * n_imp)
            b = rng.integers(0, len(imgs), 3 * n_imp)
            ok = owner[a] != owner[b]
            a, b = a[ok][:n_imp], b[ok][:n_imp]
            imp = np.einsum("ij,ij->i", E[imgs[a]], E[imgs[b]])
            r = {"people": len(people), "images": int(len(imgs)), "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp))}
            for name, t in THRESHOLDS.items():
                kf, kn = int(np.sum(gen < t)), int(np.sum(imp >= t))
                r[f"{name}/fnmr"], r[f"{name}/fnmr_ci"] = kf / max(1, len(gen)), wilson(kf, len(gen))
                r[f"{name}/fmr"], r[f"{name}/fmr_ci"] = kn / max(1, len(imp)), wilson(kn, len(imp))
                r[f"{name}/false_matches"] = kn
            res[g] = r
            print(f"{attr:9s} {g:12s} people {len(people):5d} gen {len(gen):6d} imp {len(imp):8d} | balanced FNMR "
                  f"{100 * r['balanced/fnmr']:.2f}% FMR {100 * r['balanced/fmr']:.4f}% ({r['balanced/false_matches']})")
        out["attributes"][attr] = res
    save_json("demographics", out)


if __name__ == "__main__":
    main()
