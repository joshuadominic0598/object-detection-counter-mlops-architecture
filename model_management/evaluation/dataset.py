"""
Golden test dataset helpers.

The golden dataset is the single, held-out labeled test set every trained
model is scored against, at model_management/evaluation/golden_dataset/. It
is never duplicated - training only ever sees
model_management/training/yolo_training/training_data/{train,valid} - so a
model's score can't be inflated by having trained on the same images it is
tested on.
"""

import hashlib
import os
from pathlib import Path

import yaml

EVALUATION_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_DIR = EVALUATION_DIR / "golden_dataset"

DATA_YAML = GOLDEN_DATASET_DIR / "data.yaml"
TEST_IMAGES_DIR = GOLDEN_DATASET_DIR / "test" / "images"

MISSING_DATASET_MESSAGE = (
    "Golden test dataset not found at {path}.\n"
    "Run the dataset download step in "
    "model_management/training/yolo_training/train.py (or "
    "model_management.training.yolo_training.dataset.RoboflowDatasetManager) to populate "
    "model_management/evaluation/golden_dataset/ before running an evaluation."
)


def require_golden_dataset():
    """Raise a clear, actionable error if the golden dataset hasn't been downloaded yet."""

    if not DATA_YAML.exists() or not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(MISSING_DATASET_MESSAGE.format(path=GOLDEN_DATASET_DIR))


def _dataset_files():
    require_golden_dataset()

    files = []

    # data.yaml first: it carries the class names/mapping, so a dataset
    # relabeled to different classes (e.g. car/free -> busy/free) without
    # touching a single image/label file still changes the fingerprint -
    # otherwise stale, semantically-incompatible cached results would keep
    # matching "unchanged" and get compared side by side (see model_compare).
    if DATA_YAML.exists():
        files.append(DATA_YAML)

    for relative_dir in ("test/images", "test/labels"):
        directory = GOLDEN_DATASET_DIR / relative_dir

        if not directory.exists():
            continue

        files.extend(sorted(path for path in directory.iterdir() if path.is_file()))

    return files


def _dataset_fingerprint(files):
    digest = hashlib.sha256()

    for path in files:
        digest.update(path.relative_to(GOLDEN_DATASET_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def load_dataset_version():
    """Return the golden dataset's Roboflow source metadata plus a composition fingerprint."""

    require_golden_dataset()

    data_yaml = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8")) or {}
    raw_roboflow = data_yaml.get("roboflow") or {}
    # Keep only the identifying fields - drops e.g. license, which isn't identifying info.
    source_version = (
        {key: raw_roboflow[key] for key in ("workspace", "project", "version", "url") if key in raw_roboflow}
        or None
    )

    files = _dataset_files()
    image_count = sum(1 for path in files if path.parent.name == "images")
    label_count = sum(1 for path in files if path.parent.name == "labels")
    fingerprint = _dataset_fingerprint(files)

    project = source_version.get("project") if source_version else None
    version = source_version.get("version") if source_version else None

    if not project:
        project = "golden_dataset"

    if not version:
        version = f"dataset-{fingerprint[:12]}"

    return {
        "project": project,
        "version": version,
        "fingerprint": fingerprint,
        "file_counts": {
            "images": image_count,
            "labels": label_count,
        },
        "source_version": source_version,
        "root": str(GOLDEN_DATASET_DIR),
    }


def count_test_images():
    require_golden_dataset()

    return sum(
        1
        for path in TEST_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _load_class_names():
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = data["names"]

    if isinstance(names, dict):
        return {int(index): name for index, name in names.items()}

    return dict(enumerate(names))


def resolve_eval_data_yaml():
    """
    Refresh a self-contained data.yaml pointing train/val/test at the golden
    dataset's own test split, regardless of whatever paths were baked into
    the Roboflow export this was copied from. Returns (path, class_names).
    """

    require_golden_dataset()

    class_names = _load_class_names()

    eval_data = {
        "path": str(GOLDEN_DATASET_DIR),
        "train": "test/images",
        "val": "test/images",
        "test": "test/images",
        "names": class_names,
    }

    eval_data_yaml = GOLDEN_DATASET_DIR / "eval_data.yaml"
    eval_data_yaml.write_text(yaml.safe_dump(eval_data, sort_keys=False), encoding="utf-8")

    return eval_data_yaml, class_names


def _allow_env_override():
    """Optional GOLDEN_DATASET_DIR env override, mainly useful for tests."""

    env_value = os.getenv("GOLDEN_DATASET_DIR")

    if not env_value:
        return

    global GOLDEN_DATASET_DIR, DATA_YAML, TEST_IMAGES_DIR

    GOLDEN_DATASET_DIR = Path(env_value).expanduser()
    DATA_YAML = GOLDEN_DATASET_DIR / "data.yaml"
    TEST_IMAGES_DIR = GOLDEN_DATASET_DIR / "test" / "images"


_allow_env_override()
