"""
Performance test CLI.

Validates a YOLO model directly against the golden test dataset
(model_management/evaluation/golden_dataset - see
model_management/training/yolo_training/dataset.py for how it gets populated) at every
configured confidence threshold. The actual work lives in
model_management/evaluation/evaluator.py so train.py and
model_management/orchestrator.py can call it in-process instead of shelling
out to this script.

Usage:
    PYTHONPATH=. python model_management/evaluation/scripts/run_performance_tests.py

The model under test can be swapped with the MODEL_PATH env var (a raw
file path) or the MODEL_NAME env var (a name from model_management/model_registry/registry.json,
downloaded automatically if missing - ".pt" suffixes and pasted markdown
links are tolerated):
    MODEL_PATH=tmp/models/other.pt PYTHONPATH=. python model_management/evaluation/scripts/run_performance_tests.py
    MODEL_NAME=yolo26_v2 PYTHONPATH=. python model_management/evaluation/scripts/run_performance_tests.py

Also regenerates model_comparison_dashboard.html (not just the single-model
performance_dashboard.html) so a solo test run's numbers show up there too,
without needing a separate `make model_compare`.

To test multiple registered models back-to-back and get a verdict on
which one performs better, use model_management/evaluation/dashboard/compare_models.py instead.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from model_management.evaluation.evaluator import run_for_model
from model_management.model_registry.registry import ensure_downloaded, normalize_model_name, resolve_model_path

load_dotenv()

MODEL_NAME = normalize_model_name(os.getenv("MODEL_NAME")) if os.getenv("MODEL_NAME") else None

if MODEL_NAME:
    ensure_downloaded(MODEL_NAME)
    MODEL_PATH = str(resolve_model_path(MODEL_NAME))
else:
    MODEL_PATH = os.getenv("MODEL_PATH") or str(resolve_model_path())


def open_dashboard(path):
    opener = {"darwin": "open", "win32": "start"}.get(sys.platform, "xdg-open")

    try:
        subprocess.run([opener, str(path)], check=False)
    except FileNotFoundError:
        print(f"Open the dashboard manually: {path}")


def main():
    _, _, best = run_for_model(MODEL_PATH, model_name=MODEL_NAME)
    tested_model_name = MODEL_NAME or Path(MODEL_PATH).stem

    from model_management.evaluation.dashboard.generate_performance_dashboard import (
        generate_comparison_dashboard,
        generate_dashboard,
    )

    dashboard_path = generate_dashboard(model_name=tested_model_name)
    # Keep the comparison dashboard in sync too, so a single performance_test
    # run doesn't require a separate `make model_compare` before its numbers
    # show up there.
    generate_comparison_dashboard()

    print(f"Dashboard generated at {dashboard_path}")

    open_dashboard(dashboard_path)


if __name__ == "__main__":
    main()
