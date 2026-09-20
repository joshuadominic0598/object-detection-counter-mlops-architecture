import os
from datetime import datetime

from PIL import Image

from counter.debug import draw
from counter.domain.models import CountResponse
from counter.domain.ports import ObjectDetector, ObjectCountRepo
from counter.domain.predictions import over_threshold, count

RESULTS_DIR = "results"


class PredictionService:

    def __init__(self, object_detector: ObjectDetector):
        self._object_detector = object_detector

    def get_valid_predictions(self, image, threshold, session_id):
        predictions = self._object_detector.predict(image)
        valid_predictions = list(over_threshold(predictions, threshold=threshold))

        result_image_path = self._debug_image(
            image, valid_predictions, session_id, f"valid_predictions_threshold_{threshold}"
        )

        return valid_predictions, result_image_path

    @staticmethod
    def _debug_image(image, predictions, session_id, image_name):
        if not __debug__ or image is None:
            return None

        image.seek(0)
        image = Image.open(image).convert("RGB")

        session_dir = os.path.join(RESULTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(session_dir, f"{timestamp}_{image_name}.jpg")

        draw(predictions, image, output_path)
        return output_path


class CountDetectedObjects:

    def __init__(self, object_detector: ObjectDetector, object_count_repo: ObjectCountRepo):
        self.__prediction_service = PredictionService(object_detector)
        self.__object_count_repo = object_count_repo

    def execute(self, image, threshold, session_id) -> CountResponse:
        predictions, result_image_path = self.__prediction_service.get_valid_predictions(
            image, threshold, session_id
        )

        object_counts = count(predictions)
        self.__object_count_repo.update_values(session_id, object_counts)
        total_objects = self.__object_count_repo.read_values(session_id)

        response = CountResponse(current_objects=object_counts, total_objects=total_objects)
        return response, result_image_path


class ListDetectedObjects:

    def __init__(self, object_detector: ObjectDetector):
        self.__prediction_service = PredictionService(object_detector)

    def execute(self, image, threshold, session_id):
        return self.__prediction_service.get_valid_predictions(image, threshold, session_id)