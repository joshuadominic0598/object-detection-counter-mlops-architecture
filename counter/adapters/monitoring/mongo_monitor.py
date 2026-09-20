from datetime import datetime, timezone

from pymongo import MongoClient

from counter.domain.ports import Monitor


class MongoMonitor(Monitor):

    def __init__(self, host, port, database):
        self.__client = MongoClient(host, port)
        self.__database = self.__client[database]
        self.__detection_history_col = self.__database.detection_history

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
        counts: dict,
        status_code: int,
        success: bool,
        error_message: str = None,
        result_image_path: str = None,
        counter: bool = None,
        persisted_counts: dict = None,
    ):

        document = {
            "request_id": request_id,
            "session_id": session_id,
            "request_timestamp": datetime.now(timezone.utc),
            "endpoint": endpoint,
            "image_name": image_name,
            "threshold": threshold,
            "response_time_ms": response_time_ms,
            "detected_count": detected_count,
            "detected_classes": detected_classes,
            "counts": counts,
            "status_code": status_code,
            "success": success,
            "error_message": error_message,
            "result_image_path": result_image_path,
            "counter": counter,
            "persisted_counts": persisted_counts,
        }

        self.__detection_history_col.insert_one(document)