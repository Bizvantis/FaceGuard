import os
import time

import numpy as np
import pytest

from faceguard.recognition.database import (
    FLAG_DPAPI, MAGIC, SOURCE_ADAPTIVE, FaceDatabase, Sample, decode_thumbnail,
)
from faceguard.security import dpapi
from tests.conftest import identity, samples_of


def fill(db, name="Alice", n=8, seed=0):
    rng = np.random.default_rng(seed)
    p = db.add_person(name)
    thumbs = [np.full((112, 112, 3), 120, np.uint8)]
    db.add_samples(p.id, [Sample(e, yaw=1.0) for e in samples_of(identity(rng), n, rng)], thumbs)
    return db.get(p.id)


def test_first_person_is_owner_and_names_are_unique(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    a = db.add_person("Alice")
    b = db.add_person("Bob")
    assert a.is_owner and a.trusted and not b.is_owner
    with pytest.raises(ValueError):
        db.add_person("  alice ")
    db.set_owner(b.id)
    assert db.owner().name == "Bob" and not db.get(a.id).is_owner


@pytest.mark.parametrize("encrypt", [True, False])
def test_round_trip(tmp_path, encrypt):
    path = tmp_path / "f.db"
    db = FaceDatabase(path, encrypt=encrypt)
    alice = fill(db)
    raw = path.read_bytes()
    assert raw[:4] == MAGIC
    assert bool(raw[5] & FLAG_DPAPI) == (encrypt and dpapi.available())
    if encrypt and dpapi.available():
        assert b"Alice" not in raw  # names and templates are not readable on disk
    again = FaceDatabase.open(path, encrypt=encrypt)
    p = again.find("alice")
    assert p is not None and p.count() == 8 and p.is_owner
    assert np.allclose(p.embeddings(), alice.embeddings(), atol=1e-6)
    assert decode_thumbnail(p.thumbnails[0]).shape == (112, 112, 3)


def test_snapshot_is_cached_until_change(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    fill(db)
    g1 = db.snapshot()
    assert db.snapshot() is g1 and len(g1) == 1
    fill(db, "Bob", seed=1)
    g2 = db.snapshot()
    assert g2 is not g1 and len(g2) == 2 and g2.version > g1.version


def test_damaged_file_is_moved_aside_and_backup_restored(tmp_path):
    path = tmp_path / "f.db"
    db = FaceDatabase(path)
    fill(db)
    fill(db, "Bob", seed=1)  # every save moves the previous version to .bak (one save behind)
    path.write_bytes(b"FGDB\x01\x01garbage")
    again = FaceDatabase.open(path)
    assert again.load_warning and "backup" in again.load_warning
    counts = {p.name: p.count() for p in again.persons}
    assert counts == {"Alice": 8, "Bob": 0}  # Bob's samples were added in the save that got damaged
    assert any(f.name.startswith("f.db.damaged-") for f in tmp_path.iterdir())


def test_adaptive_samples_are_rate_limited_and_capped(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    p = fill(db, n=6)
    rng = np.random.default_rng(9)
    assert db.add_adaptive_sample(p.id, Sample(identity(rng)), min_interval=600)
    assert not db.add_adaptive_sample(p.id, Sample(identity(rng)), min_interval=600)  # too soon
    for _ in range(20):
        db.get(p.id).last_adaptive = 0.0
        db.add_adaptive_sample(p.id, Sample(identity(rng)), min_interval=600)
    person = db.get(p.id)
    assert person.count(SOURCE_ADAPTIVE) == 6  # never more than the original enrollment
    assert person.count() == 12
    assert db.clear_adaptive(p.id) == 6 and db.get(p.id).count() == 6


def test_rename_delete_and_trust(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    p = fill(db)
    q = fill(db, "Bob", seed=2)
    db.rename(q.id, "Robert")
    db.set_trusted(q.id, True)
    assert db.find("robert").trusted
    with pytest.raises(ValueError):
        db.rename(q.id, "Alice")
    db.delete(p.id)
    assert [x.name for x in FaceDatabase.open(tmp_path / "f.db").persons] == ["Robert"]


def test_listeners_are_notified(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    calls = []
    db.subscribe(lambda: calls.append(time.time()))
    db.add_person("Alice")
    assert len(calls) == 1


def test_atomic_save_leaves_no_temp_files(tmp_path):
    db = FaceDatabase(tmp_path / "f.db")
    fill(db)
    fill(db, "Bob", seed=3)
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []
