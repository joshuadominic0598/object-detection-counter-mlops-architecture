"""
Runs a YOLO model through Ultralytics' validation pipeline against the
golden test dataset (model_management/evaluation/golden_dataset) at one or
more confidence thresholds, and turns the result into the same summary shape
used everywhere else in model_management (scoring.py, the dashboards,
the registry's cached performance, orchestrator.py).

This is the one place that actually runs a validation - the performance
scripts, train.py, and orchestrator.py all call `run_for_model()` rather
than duplicating this logic.

Results are written to results/model_management/evaluation/<timestamp>_<model>.json
(local only - see module docstring in model_management/orchestrator.py for why
reports never go to Drive) and, when `model_name` matches a registry entry,
the best-threshold summary is cached on that entry via
model_management.model_registry.registry.set_model_performance().
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from model_management.evaluation.dataset import count_test_images, load_dataset_version, resolve_eval_data_yaml
from model_management.evaluation.scoring import best_threshold, summarize_validation
from model_management.model_registry.registry import get_model, set_model_performance

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "model_management" / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_for_model(model_path, model_name=None, thresholds=None):
    """
    Validate `model_path` against the golden dataset at every threshold in
    `thresholds` (default DEFAULT_THRESHOLDS), save the results locally, log
    the best-threshold summary to the registry (if `model_name` is a known
    registry entry), and return (report_path, summaries, best_threshold).

    `model_name` is the registry name (when known) rather than the raw
    weights filename, so results can be matched back to a registry entry.
    """

    thresholds = thresholds or DEFAULT_THRESHOLDS

    data_yaml, class_names = resolve_eval_data_yaml()
    dataset_version = load_dataset_version()
    images_evaluated = count_test_images()
    model_name = model_name or Path(model_path).stem
    model_slug = Path(model_path).stem
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_plots_dir = PLOTS_DIR / f"{timestamp_slug}_{model_slug}"

    print(f"\nEvaluation run")
    print(f"Model           : {model_path}")
    print(f"Golden dataset  : {data_yaml.parent}")

    if dataset_version:
        print(f"Dataset version : {dataset_version.get('project')} v{dataset_version.get('version')}")

    print(f"Test images     : {images_evaluated}")
    print(f"Thresholds      : {thresholds}")

    model = YOLO(model_path)

    summaries = {}

    for threshold in thresholds:
        print(f"\nValidating at threshold {threshold}...")

        threshold_name = f"threshold_{threshold}"

        metrics = model.val(
            data=str(data_yaml),
            split="test",
            conf=threshold,
            plots=True,
            verbose=False,
            project=str(run_plots_dir),
            name=threshold_name,
            exist_ok=True,
        )

        try:
            plots_dir = str(Path(metrics.save_dir).resolve().relative_to(REPO_ROOT))
        except ValueError:
            plots_dir = str(metrics.save_dir)

        summary = summarize_validation(
            threshold=threshold,
            class_names=class_names,
            confusion_matrix=metrics.confusion_matrix.matrix,
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            images_evaluated=images_evaluated,
            plots_dir=plots_dir,
        )

        summaries[str(threshold)] = summary

    print("\nSummary")

    for threshold, summary in summaries.items():
        print(f"\nThreshold {threshold}:")
        print(f"  map50                  : {summary['map50']}")
        print(f"  map50_95               : {summary['map50_95']}")
        print(f"  overall.weighted_score : {summary['overall']['weighted_score']}")

        for class_name, class_summary in summary["per_class"].items():
            print(
                f"  {class_name:8s} precision={class_summary['precision']}, "
                f"recall={class_summary['recall']}, "
                f"pred/gt={class_summary['total_pred']}/{class_summary['total_gt']}"
            )

    best = best_threshold(summaries)

    if best is not None:
        print(f"\nBest threshold: {best} (highest macro-averaged weighted score)")

    run_report = {
        "model_path": str(model_path),
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_dataset_version": dataset_version,
        "thresholds": thresholds,
        "summary": summaries,
        "best_threshold": best,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / f"{timestamp_slug}_{model_slug}.json"

    report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print(f"\nResults saved locally to {report_path}")

    if best is not None:
        try:
            get_model(model_name)
        except KeyError:
            pass
        else:
            best_summary = summaries[best]

            try:
                results_path = str(report_path.relative_to(REPO_ROOT))
            except ValueError:
                results_path = str(report_path)

            set_model_performance(
                model_name,
                {
                    "threshold": best,
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                    "results_path": results_path,
                    "images_evaluated": best_summary.get("images_evaluated"),
                    "map50": best_summary.get("map50"),
                    "map50_95": best_summary.get("map50_95"),
                    "per_class": best_summary.get("per_class"),
                    "overall": best_summary.get("overall"),
                    "plots_dir": best_summary.get("plots_dir"),
                },
                dataset_version,
            )
            print(f"Registry updated for '{model_name}' (model_management/model_registry/registry.json)")

    return report_path, summaries, best
