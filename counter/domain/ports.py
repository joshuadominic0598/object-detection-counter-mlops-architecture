from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, List

from counter.domain.models import ObjectCount, Prediction


class ObjectDetector(ABC):

    @abstractmethod
    def predict(self, image: BinaryIO) -> List[Prediction]:
        raise NotImplementedError


class ObjectCountRepo(ABC):

    @abstractmethod
    def read_values(
        self,
        session_id: str,
        object_classes: List[str] = None,
    ) -> List[ObjectCount]:
        raise NotImplementedError

    @abstractmethod
    def update_values(
        self,
        session_id: str,
        new_values: List[ObjectCount],
    ):
        raise NotImplementedError


class Monitor(ABC):

    @abstractmethod
    def log_request(
        self,
        request_id: str,
        session_id: str,
        endpoint: str,
        image_name: str,
        threshold: float,
        response_time_ms: int,
        detected_count: int,
        detected_classes: str,
        counts: Dict[str, int],
        status_code: int,
        success: bool,
        error_message: str = None,
        result_image_path: str = None,
        counter: bool = None,
        persisted_counts: Dict[str, int] = None,
    ):
        raise NotImplementedError