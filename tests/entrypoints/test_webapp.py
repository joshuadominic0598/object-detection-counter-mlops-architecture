import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from counter.entrypoints.webapp import create_app


# Fixtures

@pytest.fixture
def client():
    app = create_app()

    with TestClient(app) as client:
        yield client


@pytest.fixture
def image_path():
    return (
        Path(__file__).parent.parent.parent
        / "resources"
        / "images"
        / "carlot.jpeg"
    )


# Single endpoint: /object-detection
# Behavior is selected via the `counter` form field:
#   counter=true  -> count objects + return cumulative totals
#   counter=false -> only list individual predictions (default)

def test_object_detection_counter_mode(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={"threshold": "0.9", "session_id": "test-session", "counter": "true"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 200

    body = response.json()

    # API contract only
    assert isinstance(body, dict)
    assert "current_objects" in body
    assert "total_objects" in body
    assert isinstance(body["current_objects"], list)
    assert isinstance(body["total_objects"], list)

    for item in body["current_objects"] + body["total_objects"]:
        assert "object_class" in item
        assert "count" in item


def test_object_detection_list_mode(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={"threshold": "0.9", "session_id": "test-session", "counter": "false"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 200

    predictions = response.json()
    assert isinstance(predictions, list)

    # Validate prediction structure only.
    # Do not make assumptions about what the model detects.
    for prediction in predictions:
        assert isinstance(prediction, dict)
        assert "class_name" in prediction
        assert "score" in prediction
        assert "box" in prediction
        assert isinstance(prediction["class_name"], str)
        assert isinstance(prediction["score"], (int, float))

        box = prediction["box"]
        assert isinstance(box, dict)
        assert "xmin" in box
        assert "ymin" in box
        assert "xmax" in box
        assert "ymax" in box


def test_object_detection_defaults_to_list_mode(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    # `counter` field omitted entirely.
    response = client.post(
        "/object-detection",
        data={"threshold": "0.9", "session_id": "test-session"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)


# Negative test: missing file

def test_object_detection_missing_file(client):
    response = client.post(
        "/object-detection",
        data={"threshold": "0.9", "session_id": "test-session"},
    )

    assert response.status_code == 400


# Negative test: invalid threshold

def test_object_detection_invalid_threshold(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={"threshold": "not-a-number", "session_id": "test-session"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 400


# Negative test: missing session id

def test_object_detection_missing_session_id(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={"threshold": "0.9"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 400


# Negative test: invalid counter flag

def test_object_detection_invalid_counter_flag(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={"threshold": "0.9", "session_id": "test-session", "counter": "maybe"},
        files={"file": ("carlot.jpeg", image, "image/jpeg")},
    )

    assert response.status_code == 400