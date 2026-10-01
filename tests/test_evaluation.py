import numpy as np
import pytest

from faceguard.recognition.analysis import analyze, auc_score, equal_error_rate, roc_curve, tar_at_far
from faceguard.recognition.database import Person, Sample
from faceguard.recognition.evaluation import evaluate_embeddings
from tests.conftest import identity, samples_of


def test_perfect_separation():
    g, i = np.full(50, 0.9), np.full(500, 0.1)
    far, tar, _ = roc_curve(g, i)
    assert auc_score(far, tar) == pytest.approx(1.0)
    assert equal_error_rate(g, i)[0] == pytest.approx(0.0)
    assert tar_at_far(g, i, 0.001)[0] == pytest.approx(1.0)


def test_overlapping_distributions():
    rng = np.random.default_rng(0)
    g, i = rng.normal(0.7, 0.1, 2000), rng.normal(0.2, 0.1, 20000)
    far, tar, _ = roc_curve(g, i)
    eer, thr = equal_error_rate(g, i)
    assert 0.99 < auc_score(far, tar) <= 1.0
    assert eer < 0.02 and 0.35 < thr < 0.55
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)


def test_evaluate_synthetic_identities():
    rng = np.random.default_rng(1)
    embs, labels = [], []
    for c in range(10):
        embs.append(samples_of(identity(rng), 8, rng))
        labels += [c] * 8
    rep = evaluate_embeddings("synthetic", np.vstack(embs), np.array(labels), threshold=0.45)
    assert rep.closed_set_accuracy == 1.0
    assert rep.open_set_dir > 0.95 and rep.open_set_far < 0.05
    assert rep.auc > 0.999 and "synthetic" in rep.summary()


def test_model_lab_analysis():
    rng = np.random.default_rng(2)
    persons = []
    for n in range(4):
        p = Person(id=str(n), name=f"P{n}")
        p.samples = [Sample(e) for e in samples_of(identity(rng), 10, rng)]
        persons.append(p)
    r = analyze(persons, 0.45)
    assert all(v == 1.0 for v in r.methods.values()) and len(r.methods) >= 3
    assert r.projection.shape == (40, 2) and r.auc > 0.999
    assert 0.36 <= r.recommended_threshold < 0.7
    one = analyze(persons[:1], 0.45)
    assert one.methods == {} and any("two people" in n for n in one.notes)
