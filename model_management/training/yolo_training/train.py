"""
Standalone training CLI: download a Roboflow dataset version, train a
model, upload the weights to Drive, and register the result in
model_management/model_registry/registry.json - nothing else. Use this when
you want a new model saved and trackable without also evaluating/promoting
it yet; use model_management/orchestrator.py instead when you also want it
evaluated against the golden dataset and considered for promotion in one
command.

--roboflow-workspace/--roboflow-project/--roboflow-version/--base-model/
--epochs/--img-size/--batch-size/--patience all default to
model_management/config.py - change that file to point every run at a new
dataset or hyperparameter set without passing flags; the flags below only
exist for one-off overrides.

Both this script and orchestrator.py import the same RoboflowDatasetManager
(dataset.py) and YoloTrainer (trainer.py), so a run behaves identically
either way - this is just a thinner driver that stops after registering.

To test this model against the golden dataset afterwards, or compare it
against every other registered model and optionally promote it, see
model_management/evaluation/scripts/run_performance_tests.py and
model_management/evaluation/dashboard/compare_models.py (`make
performance_test`/`make model_compare`).

Usage:
    PYTHONPATH=. python model_management/training/yolo_training/train.py --model-name yolo26_experiment
    PYTHONPATH=. python model_management/training/yolo_training/train.py \\
        --roboflow-workspace model-version2 --roboflow-project parking-detection-ewm7h --roboflow-version 3 \\
        --base-model yolo26n.pt --model-name yolo26_experiment \\
        --epochs 50 --img-size 640 --batch-size 8
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from model_management.config import (
    DATASET_FORMAT,
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_USE_CASE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
    USE_CASES,
)
from model_management.model_registry.drive import get_or_create_folder, upload_directory, upload_file
from model_management.model_registry.registry import add_model
from model_management.training.yolo_training.dataset import RoboflowDatasetManager
from model_management.training.yolo_training.trainer import YoloTrainer

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--roboflow-workspace", default=ROBOFLOW_WORKSPACE, help=f"Default (model_management/config.py): {ROBOFLOW_WORKSPACE}")
    dataset_group.add_argument("--roboflow-project", default=ROBOFLOW_PROJECT, help=f"Default: {ROBOFLOW_PROJECT}")
    dataset_group.add_argument("--roboflow-version", type=int, default=ROBOFLOW_VERSION, help=f"Default: {ROBOFLOW_VERSION}")
    dataset_group.add_argument("--dataset-format", default=DATASET_FORMAT)
    dataset_group.add_argument("--use-case", default=DEFAULT_USE_CASE, choices=list(USE_CASES), help=f"Which use case's required classes to validate the dataset against, and to tag the registered model with (default: {DEFAULT_USE_CASE}). See model_management/config.py.")

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    training_group.add_argument("--run-name", default=None, help="Defaults to --model-name. Folder name under results/model_management/training/runs/detect/")
    training_group.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    training_group.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    training_group.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training_group.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    registry_group = parser.add_argument_group("registry")
    registry_group.add_argument("--model-name", required=True, help="Registry name this run will be saved under.")
    registry_group.add_argument("--skip-drive-upload", action="store_true", help="Register with a local file path instead of uploading to Drive (e.g. for a local smoke test).")

    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or args.model_name

    print(f"\n1) Downloading dataset {args.roboflow_project} v{args.roboflow_version} (use case: {args.use_case})...")

    dataset_manager = RoboflowDatasetManager(
        workspace=args.roboflow_workspace,
        project=args.roboflow_project,
        version=args.roboflow_version,
        dataset_format=args.dataset_format,
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        use_case=args.use_case,
    )
    dataset = dataset_manager.download()

    print(f"\n2) Training '{args.model_name}' from {args.base_model}...")

    trainer = YoloTrainer(
        base_model=args.base_model,
        run_name=run_name,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    training_result = trainer.train(dataset.training_data_yaml)

    print(f"Training complete. Best weights: {training_result.best_weights}")

    print(f"\n3) Registering '{args.model_name}'...")

    model_file = f"{args.model_name}.pt"
    local_models_dir = REPO_ROOT / "tmp" / "models"
    local_models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = local_models_dir / model_file
    shutil.copy2(training_result.best_weights, local_model_path)

    if args.skip_drive_upload:
        drive_url = f"file://{local_model_path}"
        metrics_url = None
    else:
        parent_folder_id = get_or_create_folder("Model-training-ObjectCounter")
        run_folder_id = get_or_create_folder(run_name, parent_id=parent_folder_id)

        _, drive_url = upload_file(local_model_path, folder_id=run_folder_id)
        print(f"Model weights uploaded: {drive_url}")

        _, metrics_url = upload_directory(training_result.run_dir, folder_id=run_folder_id)
        print(f"Training run artifacts uploaded: {metrics_url}")

    add_model(args.model_name, model_file, drive_url, metrics_url=metrics_url, use_case=args.use_case)
    print(f"Registered '{args.model_name}' (use case: {args.use_case}) in model_management/model_registry/registry.json")

    print(
        "\nTo test this model against the golden dataset, run:\n"
        f"  make performance_test MODEL_NAME={args.model_name}\n"
        "To compare it against every other registered model and optionally promote it, run:\n"
        f"  make model_compare MODELS=\"{args.model_name}\" PROMOTE=1"
    )


if __name__ == "__main__":
    main()
