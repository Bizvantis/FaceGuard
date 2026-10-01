"""End-to-end tests with the real neural networks on a bundled public-domain photo."""

import cv2
import numpy as np
import pytest

from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import EngineOptions, FaceEngine
from faceguard.recognition.enrollment import samples_from_photos
from faceguard.vision.camera import ImageSource, load_image
from faceguard.vision.mesh import mediapipe_available
from tests.conftest import DATA, needs_models

pytestmark = needs_models
IMAGE = DATA / "astronaut.jpg"


@pytest.fixture(scope="module")
def engine():
    eng = FaceEngine.create()
    yield eng
    eng.close()


@pytest.fixture
def image():
    return load_image(IMAGE)


def test_detects_one_face_with_sane_landmarks(engine, image):
    faces = engine.detector.detect(image)
    assert len(faces) == 1
    f = faces[0]
    x, y, w, h = f.bbox
    assert f.score > 0.8 and 60 < w < 160
    right_eye, left_eye, nose = f.landmarks[0], f.landmarks[1], f.landmarks[2]
    assert right_eye[0] < nose[0] < left_eye[0]  # subject's right eye appears on the image's left
    assert x < nose[0] < x + w and y < nose[1] < y + h
    assert f.pose.is_frontal()


def test_embedding_is_unit_length_and_robust(engine, image):
    face = engine.detector.detect(image)[0]
    aligned, emb, q = engine.embed_face(image, face)
    assert aligned.shape == (112, 112, 3) and emb.shape == (128,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert q.ok, q.issues
    variants = [cv2.flip(image, 1), cv2.convertScaleAbs(image, alpha=0.6, beta=0), cv2.GaussianBlur(image, (5, 5), 0),
                cv2.resize(image, None, fx=0.6, fy=0.6)]
    for v in variants:
        f = engine.detector.detect(v)[0]
        assert float(engine.embed_face(v, f)[1] @ emb) > 0.8


def test_enroll_then_recognise_through_the_database(engine, image):
    db = FaceDatabase.open()
    report = samples_from_photos(engine, [IMAGE])
    assert report.used == ["astronaut.jpg"] and len(report.samples) == 1
    p = db.add_person("Eileen")
    db.add_samples(p.id, report.samples, report.thumbnails)
    engine.set_gallery(FaceDatabase.open().snapshot())
    engine.reset()
    shifted = cv2.copyMakeBorder(image, 40, 0, 60, 0, cv2.BORDER_REPLICATE)
    result = engine.process(shifted, t=0.0)
    assert result.primary.match.is_match and result.primary.match.person.name == "Eileen"
    assert result.primary.match.person.is_owner


def test_unknown_person_against_other_gallery(engine, image):
    from faceguard.recognition.gallery import Gallery, PersonInfo

    rng = np.random.default_rng(0)
    engine.set_gallery(Gallery([PersonInfo("x", "Someone")], [rng.normal(size=(5, 128))], version=12345))
    engine.reset()
    assert engine.process(image, t=0.0).primary.match.is_unknown


@pytest.mark.skipif(not mediapipe_available(), reason="mediapipe not installed")
def test_face_mesh_reads_expression(engine, image):
    result = engine.process(image, t=10.0, options=EngineOptions(mesh=True))
    mesh = result.primary.mesh
    assert mesh is not None and mesh.points.shape == (478, 2)
    assert mesh.blink < 0.3 and mesh.smile > 0.5  # eyes open, smiling
    assert abs(mesh.yaw) < 15


def test_image_source_feeds_frames():
    src = ImageSource(IMAGE, fps=30).start()
    try:
        frame, t = src.read(2.0)
        assert frame is not None and frame.shape == (320, 320, 3)
        frame2, t2 = src.read(2.0)
        assert frame2 is not None and t2 > t
    finally:
        src.stop()


def test_cli_people_and_recognize(capsys):
    from faceguard.cli import main

    assert main(["people", "list"]) == 0
    assert "Nobody is enrolled" in capsys.readouterr().out
    assert main(["recognize", "--image", str(IMAGE)]) == 0
    assert "1 face(s)" in capsys.readouterr().out
