import numpy as np

from faceguard.recognition.gallery import Gallery, Matcher, PersonInfo
from tests.conftest import identity, samples_of, unit


def make_gallery(rng, n_people=5, n=12):
    centers = [identity(rng) for _ in range(n_people)]
    persons = [PersonInfo(str(i), f"P{i}", is_owner=i == 0) for i in range(n_people)]
    return Gallery(persons, [samples_of(c, n, rng) for c in centers], version=1), centers


def test_identifies_the_right_person():
    rng = np.random.default_rng(1)
    gallery, centers = make_gallery(rng)
    matcher = Matcher(0.45, 0.30)
    for i, c in enumerate(centers):
        probe = samples_of(c, 1, rng)[0]
        m = matcher.identify(gallery, probe)
        assert m.is_match and m.person.name == f"P{i}"
        assert m.similarity > 0.6


def test_stranger_is_unknown():
    rng = np.random.default_rng(2)
    gallery, _ = make_gallery(rng)
    m = Matcher(0.45, 0.30).identify(gallery, identity(rng))
    assert m.is_unknown and m.label == "Unknown"
    assert m.person is not None  # closest person is still reported for logs


def test_in_between_is_uncertain():
    rng = np.random.default_rng(3)
    gallery, centers = make_gallery(rng, n_people=1)
    probe = unit(0.40 * centers[0] + 0.92 * identity(rng))  # cosine ~0.4 with the person
    m = Matcher(0.60, 0.20).identify(gallery, probe)
    assert m.status == "uncertain"


def test_top_k_mean_matches_manual_computation():
    rng = np.random.default_rng(4)
    gallery, centers = make_gallery(rng, n_people=2, n=7)
    probe = identity(rng)
    scores = gallery.scores(probe, k=3)
    for j, (s, e) in enumerate(gallery.slices):
        manual = np.sort(gallery.matrix[s:e] @ probe)[-3:].mean()
        assert scores[j] == np.float32(manual) or abs(scores[j] - manual) < 1e-6


def test_person_without_samples_is_not_matchable():
    rng = np.random.default_rng(5)
    g = Gallery([PersonInfo("a", "A"), PersonInfo("b", "B")], [samples_of(identity(rng), 3, rng), np.zeros((0, 128))])
    assert [p.name for p in g.persons] == ["A"]


def test_empty_gallery_reports_unknown():
    m = Matcher().identify(Gallery([], []), identity(np.random.default_rng(6)))
    assert m.is_unknown


def test_owner_and_authorized_flags():
    g = Gallery([PersonInfo("a", "A", is_owner=True), PersonInfo("b", "B", trusted=True), PersonInfo("c", "C")],
                [np.ones((1, 128))] * 3)
    assert g.owner.name == "A"
    assert [p.authorized for p in g.persons] == [True, True, False]
