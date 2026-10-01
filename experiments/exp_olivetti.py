"""AT&T / Olivetti faces (40 people x 10 low-resolution grayscale images): benchmark + classifier comparison."""

from __future__ import annotations

import numpy as np

from common import DATA, make_engine, save_json

from faceguard.recognition import evaluation as ev
from faceguard.recognition.analysis import loo_accuracy, sklearn_accuracy
from faceguard.vision.embedder import normalize


def main():
    engine = make_engine()
    home = str(DATA)
    rep = ev.evaluate_olivetti(engine, threshold=0.45, data_home=home)
    items = ev.olivetti_items(home)
    embs, labels, failures, ms = ev.embed_items(engine, items, len(items))
    embs = normalize(embs)
    methods = {**loo_accuracy(embs, labels, 3), **sklearn_accuracy(embs, labels)}
    out = {k: v for k, v in rep.__dict__.items()}
    out["classifiers"] = methods
    print(rep.summary())
    for k, v in methods.items():
        print(f"  {k:26s} {100 * v:.2f}%")
    save_json("olivetti", out)


if __name__ == "__main__":
    main()
