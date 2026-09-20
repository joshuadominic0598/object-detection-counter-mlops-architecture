import io
import json
from pathlib import Path

from counter.entrypoints.webapp import create_app


class TestObjectDetectionE2E:

    def setup_method(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.image_path = Path(__file__).parent.parent.parent / "resources" / "images" / "carlot.jpeg"

    def test_object_detection_counter_mode(self):
        with self.app.test_client() as client, open(self.image_path, "rb") as f:
            response = client.post(
                "/object-detection",
                data={
                    "threshold": "0.5",
                    "session_id": "test-session",
                    "counter": "true",
                    "file": (io.BytesIO(f.read()), "carlot.jpeg"),
                },
                content_type="multipart/form-data",
            )

        assert response.status_code == 200

        body = json.loads(response.data)
        assert isinstance(body, dict)
        assert "current_objects" in body
        assert "total_objects" in body
        assert isinstance(body["current_objects"], list)
        assert isinstance(body["total_objects"], list)

    def test_object_detection_list_mode(self):
        with self.app.test_client() as client, open(self.image_path, "rb") as f:
            response = client.post(
                "/object-detection",
                data={
                    "threshold": "0.5",
                    "session_id": "test-session",
                    "counter": "false",
                    "file": (io.BytesIO(f.read()), "carlot.jpeg"),
                },
                content_type="multipart/form-data",
            )

        assert response.status_code == 200

        predictions = json.loads(response.data)
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