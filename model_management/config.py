"""
Single source of truth for model_management: which use cases exist and the
classes each one requires, which Roboflow dataset a run trains/evaluates
against by default, and the default training hyperparameters.

To point every `make train`/`make orchestrate` run at a new dataset version,
change ROBOFLOW_* here - that's the one place it needs to change. CLI flags
on train.py/orchestrator.py still exist for one-off experiments, but they all
default to the values below, so a plain `make train`/`make orchestrate` with
no ARGS always uses this config.
"""

USE_CASES = {
    "carpark": {
        "classes": ["Car", "Free"],
        "description": "Parking space occupancy - counts occupied (Car) vs empty (Free) spots.",
    },
}

DEFAULT_USE_CASE = "carpark"

# Roboflow dataset location - replace these with your own workspace/project/version
# before running train/orchestrate. See https://roboflow.com.
ROBOFLOW_WORKSPACE = "your-roboflow-workspace"
ROBOFLOW_PROJECT = "your-roboflow-project"
ROBOFLOW_VERSION = 1
DATASET_FORMAT = "yolo26"

# Training hyperparameter defaults.
DEFAULT_BASE_MODEL = "yolo26l.pt"
DEFAULT_IMG_SIZE = 416
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 8
DEFAULT_PATIENCE = 25


def get_use_case(name):
    if name not in USE_CASES:
        raise KeyError(f"Unknown use case '{name}'. Known use cases: {', '.join(USE_CASES)}")

    return USE_CASES[name]


def expected_classes(name):
    return get_use_case(name)["classes"]


def _normalized(names):
    return sorted(str(name).strip().lower() for name in names)


def validate_classes(use_case_name, class_names):
    """Hard stop: raise if `class_names` (an iterable of names, or a
    {index: name} dict) don't match `use_case_name`'s expected classes,
    case-insensitively. Called before any dataset files are written so a
    mismatched dataset never partially overwrites training_data/golden_dataset.
    """

    names = class_names.values() if isinstance(class_names, dict) else class_names
    expected = expected_classes(use_case_name)

    if _normalized(names) != _normalized(expected):
        raise RuntimeError(
            f"Dataset classes {sorted(names)} do not match the '{use_case_name}' use case's "
            f"required classes {expected}. Refusing to overwrite training_data/golden_dataset. "
            f"Pass --use-case for a different use case, or add a new one in model_management/config.py."
        )
