import io
from pathlib import Path

from fastapi.testclient import TestClient

from counter.entrypoints.webapp import create_app


class TestObjectDetectionE2E:

    def setup_method(self):
        self.app = create_app()
        self.image_path = Path(__file__).parent.parent.parent / "resources" / "images" / "carlot.jpeg"

    def test_object_detection_counter_mode(self):
        with TestClient(self.app) as client, open(self.image_path, "rb") as f:
            response = client.post(
                "/object-detection",
                data={"threshold": "0.5", "session_id": "test-session", "counter": "true"},
                files={"file": ("carlot.jpeg", io.BytesIO(f.read()), "image/jpeg")},
            )

        assert response.status_code == 200

        body = response.json()
        assert isinstance(body, dict)
        assert "current_objects" in body
        assert "total_objects" in body
        assert isinstance(body["current_objects"], list)
        assert isinstance(body["total_objects"], list)

    def test_object_detection_list_mode(self):
        with TestClient(self.app) as client, open(self.image_path, "rb") as f:
            response = client.post(
                "/object-detection",
                data={"threshold": "0.5", "session_id": "test-session", "counter": "false"},
                files={"file": ("carlot.jpeg", io.BytesIO(f.read()), "image/jpeg")},
            )

        assert response.status_code == 200

        predictions = response.json()
        assert isinstance(predictions, list)

        for prediction in predictions:
            assert "class_name" in prediction
            assert "score" in prediction
            assert "box" in prediction
            assert isinstance(prediction["class_name"], str)
            assert isinstance(prediction["score"], (int, float))

            box = prediction["box"]
            assert "xmin" in box
            assert "ymin" in box
            assert "xmax" in box
            assert "ymax" in box