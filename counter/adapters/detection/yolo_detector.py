from typing import BinaryIO, List

from PIL import Image
from ultralytics import YOLO

from counter.domain.models import Box, Prediction
from counter.domain.ports import ObjectDetector


class YOLODetector(ObjectDetector):

    def __init__(self, model_path: str):
        self.__model = YOLO(model_path)

    def predict(self, image: BinaryIO) -> List[Prediction]:
        image.seek(0)
        pil_image = Image.open(image).convert("RGB")

        results = self.__model.predict(source=pil_image, verbose=False)

        predictions = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

                predictions.append(
                    Prediction(
                        class_name=result.names[class_id],
                        score=score,
                        box=Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    )
                )

        return predictions