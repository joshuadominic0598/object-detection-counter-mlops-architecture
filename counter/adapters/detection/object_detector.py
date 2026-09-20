from typing import BinaryIO, List

from counter.domain.models import Box, Prediction
from counter.domain.ports import ObjectDetector


class FakeObjectDetector(ObjectDetector):

    def predict(self, image: BinaryIO) -> List[Prediction]:
        return [
            Prediction(
                class_name="cat",
                score=0.999190748,
                box=Box(
                    xmin=0.367288858,
                    ymin=0.278333426,
                    xmax=0.735821366,
                    ymax=0.6988855,
                ),
            ),
        ]