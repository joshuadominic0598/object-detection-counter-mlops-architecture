import os

from dotenv import load_dotenv

from counter.adapters.database.count_mongo_repo import CountMongoDBRepo
from counter.adapters.detection.yolo_detector import YOLODetector
from counter.adapters.monitoring.mongo_monitor import MongoMonitor
from counter.domain.actions import CountDetectedObjects, ListDetectedObjects

load_dotenv()

# Active model resolution
try:
    from model_management.model_registry.registry import resolve_model_path
    _ACTIVE_MODEL_PATH = str(resolve_model_path())
except Exception:
    # Registry missing/unreadable - fall back to the historical default
    _ACTIVE_MODEL_PATH = "tmp/models/yolo26.pt"

MODEL_PATH = os.getenv("MODEL_PATH") or _ACTIVE_MODEL_PATH

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB = os.getenv("MONGO_DB", "prod_counter")


# Dependency wiring

def get_detector():
    return YOLODetector(MODEL_PATH)


def get_repo():
    return CountMongoDBRepo(host=MONGO_HOST, port=MONGO_PORT, database=MONGO_DB)


def prod_count_action():
    return CountDetectedObjects(get_detector(), get_repo())


def prod_object_list_action():
    return ListDetectedObjects(get_detector())


def get_count_action():
    return prod_count_action()


def get_object_list_action():
    return prod_object_list_action()


def get_monitor():
    return MongoMonitor(host=MONGO_HOST, port=MONGO_PORT, database=MONGO_DB)