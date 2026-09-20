"""
End-to-end pipeline: download a dataset, train a model, register it,
evaluate it against the golden test dataset, and decide whether it should
be promoted to active - all from one command. Use
model_management/training/yolo_training/train.py instead if you just want
to train + register (no evaluation/promotion).

Only model weights and training run artifacts (results.csv, plots,
args.yaml) go to Google Drive - see model_management/model_registry/drive.py.
Performance/evaluation reports are generated locally only (see
model_management/evaluation/evaluator.py) and logged to the registry, so
finding a model later only ever requires the registry entry.

--roboflow-workspace/--roboflow-project/--roboflow-version/--base-model/
--epochs/--img-size/--batch-size/--patience all default to
model_management/config.py - change that file to point every run at a new
dataset or hyperparameter set without passing flags; the flags below only
exist for one-off overrides.

--use-case (default: carpark) picks which model_management/config.py
entry the downloaded dataset's classes must match - a mismatch is a hard
stop before any file is overwritten - and tags the registered model with
that use case, so future use cases can be added without touching this file.

If the downloaded golden test dataset turns out to differ from the one
every other registered model was last scored against (a version bump, or a
relabeling that changes the class names), every registered model is
re-tested and ranked together via model_management/evaluation/dashboard/compare_models.py's
run_comparison() - not just this run's candidate vs the current active
model - so stale results are never compared side by side with fresh ones.
Otherwise it's a plain candidate-vs-active comparison.

Promotion is never hard-coded to one metric or one class: --promote-metric
picks precision, recall, or weighted_score (F1, macro-averaged across
whatever classes the model has - default), and --auto-promote decides
whether crossing that bar actually flips the active model or just prints a
recommendation. See model_management/promotion.py for the comparison logic
this shares with model_management/evaluation/dashboard/compare_models.py.

Usage:
    PYTHONPATH=. python model_management/orchestrator.py --model-name yolo26_v7
    PYTHONPATH=. python model_management/orchestrator.py \\
        --roboflow-workspace model-version2 --roboflow-project parking-detection-ewm7h --roboflow-version 3 \\
        --base-model yolo26n.pt --model-name yolo26_v7 --use-case carpark \\
        --epochs 50 --img-size 640 --batch-size 8 \\
        --promote-metric weighted_score --auto-promote
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
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
from model_management.evaluation.dashboard.compare_models import run_comparison
from model_management.evaluation.dataset import load_dataset_version
from model_management.evaluation.evaluator import run_for_model
from model_management.model_registry.drive import get_or_create_folder, upload_directory, upload_file
from model_management.model_registry.registry import (
    add_model,
    get_active_model_name,
    get_model_performance,
    resolve_model_path,
)
from model_management.promotion import DEFAULT_METRIC, METRICS, candidate_beats_active, metric_value
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
    training_group.add_argument("--run-name", default=None, help="Defaults to --model-name.")
    training_group.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    training_group.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    training_group.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training_group.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    registry_group = parser.add_argument_group("registry")
    registry_group.add_argument("--model-name", required=True, help="Registry name this run will be saved under.")
    registry_group.add_argument("--skip-drive-upload", action="store_true", help="Register with a local file path instead of uploading to Drive (e.g. for a local smoke test).")

    promote_group = parser.add_argument_group("promotion")
    promote_group.add_argument("--promote-metric", choices=METRICS, default=DEFAULT_METRIC, help=f"Metric compared against the active model (default: {DEFAULT_METRIC}).")
    promote_group.add_argument("--auto-promote", action="store_true", help="Actually flip the active model if the candidate wins on --promote-metric; otherwise only print a recommendation.")

    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or args.model_name

    try:
        previous_fingerprint = load_dataset_version()["fingerprint"]
    except FileNotFoundError:
        previous_fingerprint = None

    print(f"\n1) Downloading dataset {args.roboflow_project} v{args.roboflow_version}...")

    dataset_manager = RoboflowDatasetManager(
        workspace=args.roboflow_workspace,
        project=args.roboflow_project,
        version=args.roboflow_version,
        dataset_format=args.dataset_format,
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        use_case=args.use_case,
    )
    dataset = dataset_manager.download()

    dataset_changed = previous_fingerprint != load_dataset_version()["fingerprint"]

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

    print(f"\n4) Evaluating '{args.model_name}' against the golden test dataset...")

    _, summaries, best = run_for_model(str(local_model_path), model_name=args.model_name)

    if best is None:
        print("Evaluation produced no usable result - skipping promotion check.")
        return

    candidate_summary = summaries[best]

    if dataset_changed:
        print(
            f"\n5) Golden test dataset changed since the last run - re-comparing every "
            f"registered model (not just '{args.model_name}') on {args.promote_metric}..."
        )
        run_comparison(metric=args.promote_metric, promote=args.auto_promote, show_dashboard=False)
        return

    print(f"\n5) Comparing '{args.model_name}' against the active model on {args.promote_metric}...")

    active_name = get_active_model_name()
    active_performance = get_model_performance(active_name)
    candidate_value = metric_value(candidate_summary, args.promote_metric)
    active_value = metric_value(active_performance, args.promote_metric)

    def fmt(value):
        return f"{value:.1f}%" if value is not None else "N/A"

    print(f"  {args.model_name}: {args.promote_metric} = {fmt(candidate_value)}")
    print(f"  {active_name} (active): {args.promote_metric} = {fmt(active_value)}")

    if args.model_name == active_name:
        print(f"\n'{args.model_name}' is already the active model.")
    elif candidate_beats_active(candidate_summary, active_performance, args.promote_metric):
        print(f"\nCandidate '{args.model_name}' beats active model '{active_name}' on {args.promote_metric}.")

        if args.auto_promote:
            from model_management.model_registry.registry import set_active_model

            set_active_model(args.model_name)
            print(f"Registry updated: active model is now '{args.model_name}'.")
        else:
            print("Re-run with --auto-promote, or `make model_promote MODEL=" + args.model_name + "`, to make it active.")
    else:
        print(f"\nCandidate '{args.model_name}' does not beat active model '{active_name}' on {args.promote_metric}. Not promoting.")


if __name__ == "__main__":
    main()
