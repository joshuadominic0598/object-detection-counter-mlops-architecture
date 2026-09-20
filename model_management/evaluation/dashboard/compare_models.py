"""
Tests every registered, downloaded model (or a chosen subset) against the
golden test dataset and reports each model's metrics side by side, so a new
candidate model can be compared against the model currently in production
before switching to it.

This only reports numbers - see model_management/promotion.py for the
metric comparison used when --promote is passed, and
model_management/orchestrator.py for the full train -> evaluate -> promote
pipeline. There is no hard-coded "best fit" rule baked into this script or
the dashboards; --metric picks which number decides promotion, and that
choice is yours to make per run.

Results are cached on each model's registry entry (see
model_management/model_registry/registry.py's set_model_performance/
get_model_performance) keyed to the golden dataset's version, so re-running
this script doesn't re-test a model unless the golden dataset has changed
or --retest is passed.

run_comparison() is the reusable half of this module - orchestrator.py calls
it directly (instead of shelling out to this CLI) whenever a training run's
dataset version turns out to differ from the one every other registered
model was last tested against, so a dataset change always re-compares every
model, not just the one just trained.

Usage:
    PYTHONPATH=. python model_management/evaluation/dashboard/compare_models.py
    PYTHONPATH=. python model_management/evaluation/dashboard/compare_models.py --models yolo26 yolo26_v2
    PYTHONPATH=. python model_management/evaluation/dashboard/compare_models.py --promote --metric recall
    PYTHONPATH=. python model_management/evaluation/dashboard/compare_models.py --retest
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from model_management.evaluation.dataset import load_dataset_version
from model_management.evaluation.evaluator import run_for_model
from model_management.model_registry.registry import (
    ensure_downloaded,
    get_active_model_name,
    get_model_performance,
    is_downloaded,
    list_models,
    normalize_model_name,
    resolve_model_path,
    set_active_model,
)
from model_management.evaluation.dashboard.generate_performance_dashboard import generate_comparison_dashboard
from model_management.promotion import DEFAULT_METRIC, METRICS, candidate_beats_active, metric_value, rank_models


def open_dashboard(path):
    opener = {"darwin": "open", "win32": "start"}.get(sys.platform, "xdg-open")

    try:
        subprocess.run([opener, str(path)], check=False)
    except FileNotFoundError:
        print(f"Open the dashboard manually: {path}")


def best_summary_from_cache(performance):
    """Reconstruct the single best-threshold summary dict a cached registry
    entry stores, in the same shape run_for_model()'s summaries values use."""

    return {
        "threshold": performance["threshold"],
        "images_evaluated": performance.get("images_evaluated"),
        "map50": performance.get("map50"),
        "map50_95": performance.get("map50_95"),
        "per_class": performance.get("per_class", {}),
        "overall": performance.get("overall", {}),
        "plots_dir": performance.get("plots_dir"),
    }


def get_or_run(name, dataset_version, retest):
    """Return (report_path, best_summary) for `name`, reusing the registry's
    cached result when it matches the current golden dataset composition."""

    cached = None if retest else get_model_performance(name, dataset_version)

    if cached is not None:
        print(f"\nUsing cached result for '{name}' (golden dataset composition unchanged).")
        return cached.get("results_path"), best_summary_from_cache(cached)

    ensure_downloaded(name)
    model_path = str(resolve_model_path(name))

    print(f"\nTesting model '{name}' ({model_path})")

    report_path, summaries, best = run_for_model(model_path, model_name=name)

    if best is None:
        return report_path, None

    return report_path, summaries[best]


def run_comparison(names=None, metric=DEFAULT_METRIC, promote=False, retest=False, show_dashboard=True):
    """Test `names` (default: every downloaded model, or every registered
    model if none are downloaded) against the golden dataset, print a ranked
    table, optionally promote the top-ranked model, and refresh the
    comparison dashboard. Returns (ranked_names, summaries).

    Shared by this script's CLI and model_management/orchestrator.py, which
    calls this instead of duplicating comparison logic whenever the golden
    dataset version changed during its own run.
    """

    if names:
        names = [normalize_model_name(name) for name in names]
    else:
        names = [name for name in list_models() if is_downloaded(name)]

        if not names:
            print("No downloaded models found in tmp/models - falling back to every registered model.")
            names = list(list_models())

    active_name = get_active_model_name()
    dataset_version = load_dataset_version()

    print(f"\nModel comparison — models: {names}, active: {active_name}, metric: {metric}")

    summaries = {}

    for name in names:
        _, summary = get_or_run(name, dataset_version, retest)
        summaries[name] = summary

    scored_names = [name for name in names if summaries[name] is not None]

    if not scored_names:
        print("No model produced a usable result - check the golden dataset and model files.")
        return [], summaries

    print(f"\nRanked by {metric}:")

    ranked = rank_models(summaries, metric)

    for name in ranked:
        if name not in scored_names:
            continue

        value = metric_value(summaries[name], metric)
        marker = " (active)" if name == active_name else ""
        value_text = f"{value:.1f}%" if value is not None else "N/A"
        threshold_text = summaries[name]["threshold"] if summaries[name] else "N/A"
        print(f"  {name}{marker}: threshold {threshold_text}, {metric} {value_text}")

    top_name = ranked[0]
    active_summary = summaries.get(active_name)

    if top_name == active_name:
        print(f"\n'{active_name}' remains the top-ranked model on {metric}.")
    elif candidate_beats_active(summaries[top_name], active_summary, metric):
        print(f"\nCandidate '{top_name}' outperforms the active model '{active_name}' on {metric}.")

        if promote:
            set_active_model(top_name)
            print(f"Registry updated: active model is now '{top_name}'.")
        else:
            print(f"Re-run with --promote to make '{top_name}' the active model.")
    else:
        print(f"\nNo candidate outperforms the active model '{active_name}' on {metric}.")

    dashboard_path = generate_comparison_dashboard(model_names=names)

    print(f"\nDashboard generated at {dashboard_path}")

    if show_dashboard:
        open_dashboard(dashboard_path)

    return ranked, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Registry model names to test (default: every downloaded model in tmp/models).")
    parser.add_argument("--metric", choices=METRICS, default=DEFAULT_METRIC, help=f"Metric used to rank models and decide --promote (default: {DEFAULT_METRIC}).")
    parser.add_argument("--promote", action="store_true", help="Set the top-ranked model (by --metric) as the registry's active model.")
    parser.add_argument("--retest", action="store_true", help="Ignore cached registry results and re-run every model.")
    args = parser.parse_args()

    run_comparison(names=args.models, metric=args.metric, promote=args.promote, retest=args.retest)


if __name__ == "__main__":
    main()
