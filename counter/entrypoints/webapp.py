from io import BytesIO
import time
import uuid

from flask import Flask, request, jsonify, abort

from counter import config
from counter.monitoring.request_monitor import RequestMonitor


# Request parsing

def get_request_data():
    request_id = str(uuid.uuid4())
    start_time = time.time()

    session_id = request.form.get("session_id")
    if not session_id:
        abort(400, description="session_id is required")

    threshold_raw = request.form.get("threshold", 0.5)
    try:
        threshold = float(threshold_raw)
    except (ValueError, TypeError):
        abort(400, description="threshold must be a valid float")

    # counter=true counts objects and persists totals, counter=false only lists predictions
    counter_raw = request.form.get("counter", "false")
    counter_flag = parse_bool(counter_raw)

    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        abort(400, description="file is required")
    if not uploaded_file.filename:
        abort(400, description="file must have a filename")

    image_name = uploaded_file.filename
    image = BytesIO()
    uploaded_file.save(image)
    image.seek(0)

    return request_id, session_id, start_time, threshold, counter_flag, image_name, image


def parse_bool(value):
    normalized = str(value).strip().lower()

    if normalized == "true":
        return True
    if normalized == "false":
        return False

    abort(400, description="counter must be 'true' or 'false'")


# Application setup

def create_app():
    app = Flask(__name__)

    count_action = config.get_count_action()
    object_list_action = config.get_object_list_action()
    monitor = RequestMonitor(config.get_monitor())

    # Routes

    @app.route("/object-detection", methods=["POST"])
    def object_detection():
        (
            request_id,
            session_id,
            start_time,
            threshold,
            counter_flag,
            image_name,
            image,
        ) = get_request_data()

        try:
            if counter_flag:
                count_response, result_image_path = count_action.execute(image, threshold, session_id)
                counts = monitor.build_counts(count_response.current_objects)
                persisted_counts = monitor.build_counts(count_response.total_objects)
                response_body = count_response
            else:
                predictions, result_image_path = object_list_action.execute(image, threshold, session_id)
                counts = monitor.build_counts_from_predictions(predictions)
                persisted_counts = None
                response_body = predictions

            monitor.log_success(
                request_id=request_id,
                session_id=session_id,
                endpoint="/object-detection",
                image_name=image_name,
                threshold=threshold,
                start_time=start_time,
                counts=counts,
                result_image_path=result_image_path,
                counter=counter_flag,
                persisted_counts=persisted_counts,
            )

            return jsonify(response_body)

        except Exception as ex:
            monitor.log_failure(
                request_id=request_id,
                session_id=session_id,
                endpoint="/object-detection",
                image_name=image_name,
                threshold=threshold,
                start_time=start_time,
                error_message=str(ex),
                counter=counter_flag,
            )
            raise

    return app


if __name__ == "__main__":
    app = create_app()
    app.run("0.0.0.0", debug=True)