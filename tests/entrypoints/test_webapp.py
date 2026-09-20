import io
import json
from pathlib import Path

import pytest

from counter.entrypoints.webapp import create_app


# Fixtures

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
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
        data={
            "threshold": "0.9",
            "session_id": "test-session",
            "counter": "true",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 200

    body = json.loads(response.data)

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
        data={
            "threshold": "0.9",
            "session_id": "test-session",
            "counter": "false",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 200

    predictions = json.loads(response.data)
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
        data={
            "threshold": "0.9",
            "session_id": "test-session",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 200
    assert isinstance(json.loads(response.data), list)


# Negative test: missing file

def test_object_detection_missing_file(client):
    response = client.post(
        "/object-detection",
        data={
            "threshold": "0.9",
            "session_id": "test-session",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


# Negative test: invalid threshold

def test_object_detection_invalid_threshold(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={
            "threshold": "not-a-number",
            "session_id": "test-session",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 400


# Negative test: missing session id

def test_object_detection_missing_session_id(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={
            "threshold": "0.9",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 400


# Negative test: invalid counter flag

def test_object_detection_invalid_counter_flag(client, image_path):
    with open(image_path, "rb") as f:
        image = io.BytesIO(f.read())

    response = client.post(
        "/object-detection",
        data={
            "threshold": "0.9",
            "session_id": "test-session",
            "counter": "maybe",
            "file": (image, "carlot.jpeg"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 400