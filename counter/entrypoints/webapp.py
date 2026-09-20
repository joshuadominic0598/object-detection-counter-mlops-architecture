from io import BytesIO
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from counter import config
from counter.monitoring.request_monitor import RequestMonitor


def parse_bool(value):
    normalized = str(value).strip().lower()

    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise HTTPException(status_code=400, detail="counter must be 'true' or 'false'")


# Application setup

def create_app():
    app = FastAPI()

    count_action = config.get_count_action()
    object_list_action = config.get_object_list_action()
    monitor = RequestMonitor(config.get_monitor())

    # FastAPI/Pydantic reports missing/malformed form fields as 422 by
    # default - map that back to 400 to keep the existing API contract.
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request, exc):
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    # Routes

    @app.post("/object-detection")
    async def object_detection(
        session_id: str = Form(...),
        threshold: float = Form(0.5),
        counter: str = Form("false"),
        file: UploadFile = File(...),
    ):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # counter=true counts objects and persists totals, counter=false only lists predictions
        counter_flag = parse_bool(counter)

        if not file.filename:
            raise HTTPException(status_code=400, detail="file must have a filename")

        image_name = file.filename
        image = BytesIO(await file.read())

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

            return response_body

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
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=5000)