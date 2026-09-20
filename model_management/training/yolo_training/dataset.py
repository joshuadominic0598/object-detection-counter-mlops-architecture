"""
Downloads a dataset version from Roboflow and arranges it into the two
places the rest of model_management expects:

- model_management/training/yolo_training/training_data/{train,valid} - used
  for training only (see trainer.py). Never contains the golden test split,
  so a model can't be scored against images it was trained on.
- model_management/evaluation/golden_dataset - Roboflow's own `test` split,
  used by every future trained model's evaluation (see
  model_management/evaluation/). Re-downloading here replaces the previous
  copy; dataset versioning itself stays managed on Roboflow, its golden
  data.yaml's own `roboflow:` block records which Roboflow version is
  currently in place.

Used directly by model_management/orchestrator.py and model_management/training/yolo_training/train.py.
Lives inside yolo_training/ (rather than training/ directly) since it's
YOLO/Roboflow-specific - a future training approach (e.g. a
tensorflow_coco_training/ package) would bring its own.

Before anything is copied, the downloaded dataset's class names are checked
against `use_case`'s required classes (model_management/config.py) - a
mismatch raises instead of overwriting training_data/golden_dataset, so a
carpark model can never be accidentally retrained on a different use case's
classes. The raw Roboflow export directory is deleted once its contents are
copied out, so a download never leaves a stray extracted-dataset folder
behind.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from model_management.config import DEFAULT_USE_CASE, validate_classes

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_DIR = REPO_ROOT / "model_management" / "training" / "yolo_training" / "training_data"
GOLDEN_DATASET_DIR = REPO_ROOT / "model_management" / "evaluation" / "golden_dataset"


@dataclass
class DatasetPaths:
    training_data_yaml: Path
    training_data_dir: Path
    golden_dataset_dir: Path
    class_names: dict


def _load_class_names(data_yaml):
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = data["names"]

    return {int(index): name for index, name in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def _replace_dir(destination, source):
    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)


class RoboflowDatasetManager:
    """Downloads one Roboflow dataset version and splits it into the
    training_data/ (train+valid) and evaluation/golden_dataset (test)
    directories every other model_management module reads from."""

    def __init__(self, workspace, project, version, dataset_format="yolo26", api_key=None, use_case=DEFAULT_USE_CASE):
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dataset_format = dataset_format
        self.api_key = api_key
        self.use_case = use_case

    def download(self):
        from roboflow import Roboflow

        if not self.api_key:
            raise RuntimeError("Set ROBOFLOW_API_KEY in your .env file before downloading a dataset.")

        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        version = project.version(self.version)
        dataset = version.download(self.dataset_format)

        source_root = Path(dataset.location)
        source_data_yaml = source_root / "data.yaml"
        class_names = _load_class_names(source_data_yaml)

        # Hard stop before touching training_data/golden_dataset: a dataset
        # whose classes don't match this use case's requirement must never
        # partially overwrite them (see model_management/config.py).
        validate_classes(self.use_case, class_names)

        project_id = project.id.split("/")[-1] if "/" in project.id else project.id

        # Train + valid -> training_data/, used for training only.
        _replace_dir(TRAINING_DATA_DIR / "train", source_root / "train")
        _replace_dir(TRAINING_DATA_DIR / "valid", source_root / "valid")

        training_data_yaml = TRAINING_DATA_DIR / "data.yaml"
        training_data_yaml.write_text(
            yaml.safe_dump(
                {
                    "train": "train/images",
                    "val": "valid/images",
                    "nc": len(class_names),
                    "names": class_names,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        # Test -> evaluation/golden_dataset, the one and only golden test set.
        test_split = source_root / "test"

        if not test_split.exists():
            raise FileNotFoundError(f"No test split found in downloaded dataset at {source_root}")

        _replace_dir(GOLDEN_DATASET_DIR / "test", test_split)

        golden_data_yaml = GOLDEN_DATASET_DIR / "data.yaml"
        golden_data_yaml.write_text(
            yaml.safe_dump(
                {
                    "train": "test/images",
                    "val": "test/images",
                    "test": "test/images",
                    "nc": len(class_names),
                    "names": class_names,
                    # No license - it's not identifying info, unlike the fields below.
                    "roboflow": {
                        "workspace": self.workspace,
                        "project": project_id,
                        "version": version.version,
                        "url": f"https://universe.roboflow.com/{self.workspace}/{project_id}/dataset/{version.version}",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        # Roboflow's raw export directory is now fully copied into
        # training_data/ and golden_dataset/ - delete it so a download never
        # leaves a stray extracted-dataset folder (e.g. "Demo2-2/") behind.
        shutil.rmtree(source_root, ignore_errors=True)

        return DatasetPaths(
            training_data_yaml=training_data_yaml,
            training_data_dir=TRAINING_DATA_DIR,
            golden_dataset_dir=GOLDEN_DATASET_DIR,
            class_names=class_names,
        )
