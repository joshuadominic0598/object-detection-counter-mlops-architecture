import time

from counter.domain.predictions import count


class RequestMonitor:

    def __init__(self, monitor):
        self.__monitor = monitor

    def log_success(
        self,
        request_id,
        session_id,
        endpoint,
        image_name,
        threshold,
        start_time,
        counts,
        result_image_path=None,
        counter=None,
        persisted_counts=None,
    ):
        self.__monitor.log_request(
            request_id=request_id,
            session_id=session_id,
            endpoint=endpoint,
            image_name=image_name,
            threshold=threshold,
            response_time_ms=self._get_elapsed_ms(start_time),
            detected_count=sum(counts.values()),
            detected_classes=",".join(sorted(counts.keys())),
            counts=counts,
            status_code=200,
            success=True,
            error_message=None,
            result_image_path=result_image_path,
            counter=counter,
            persisted_counts=persisted_counts,
        )

    def log_failure(
        self,
        request_id,
        session_id,
        endpoint,
        image_name,
        threshold,
        start_time,
        error_message,
        status_code=500,
        counter=None,
    ):
        self.__monitor.log_request(
            request_id=request_id,
            session_id=session_id,
            endpoint=endpoint,
            image_name=image_name,
            threshold=threshold,
            response_time_ms=self._get_elapsed_ms(start_time),
            detected_count=0,
            detected_classes="",
            counts={},
            status_code=status_code,
            success=False,
            error_message=error_message,
            result_image_path=None,
            counter=counter,
            persisted_counts=None,
        )

    @staticmethod
    def build_counts(object_counts):
        return {item.object_class: item.count for item in object_counts}

    @staticmethod
    def build_counts_from_predictions(predictions):
        return {item.object_class: item.count for item in count(predictions)}

    @staticmethod
    def _get_elapsed_ms(start_time):
        return int((time.time() - start_time) * 1000)