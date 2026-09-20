"""
Thin wrapper around Ultralytics training/validation so the notebook and
model_management/orchestrator.py share one implementation instead of the
notebook inlining `model.train(...)` directly.

Augmentation/optimizer defaults here match what was tuned by hand in the
original notebook; they're kept as YoloTrainer's defaults so a run in
orchestrator.py behaves the same as a notebook run unless overridden. Lives
inside yolo_training/ (rather than training/ directly) since it's a YOLO-
specific wrapper - a future training approach would bring its own trainer.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "results" / "model_management" / "training" / "runs" / "detect"


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


@dataclass
class TrainingResult:
    run_dir: Path
    best_weights: Path
    last_weights: Path
    results_csv: Path


class YoloTrainer:
    def __init__(
        self,
        base_model,
        run_name,
        img_size=416,
        epochs=50,
        batch_size=8,
        patience=25,
        device=None,
    ):
        self.base_model = base_model
        self.run_name = run_name
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or detect_device()

    def train(self, data_yaml):
        run_dir = (RUNS_DIR / self.run_name).resolve()
        run_dir.parent.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.base_model)

        model.train(
            data=str(data_yaml),
            imgsz=self.img_size,
            epochs=self.epochs,
            batch=self.batch_size,
            device=self.device,
            workers=2,
            pretrained=True,
            patience=self.patience,
            cache=False,
            save=True,
            save_period=10,
            val=True,
            degrees=5,
            translate=0.05,
            scale=0.20,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.25,
            project=str(run_dir.parent),
            name=run_dir.name,
            seed=42,
            plots=True,
            verbose=True,
        )

        return TrainingResult(
            run_dir=run_dir,
            best_weights=run_dir / "weights" / "best.pt",
            last_weights=run_dir / "weights" / "last.pt",
            results_csv=run_dir / "results.csv",
        )

    def validate(self, weights_path, data_yaml, img_size=None, batch_size=None):
        """Independent test-split validation, separate from the val split
        used during training - same golden_dataset test evaluation later
        does with a fresh model.val() call, but at training resolution."""

        model = YOLO(str(weights_path))

        return model.val(
            data=str(data_yaml),
            split="test",
            imgsz=img_size or self.img_size,
            batch=batch_size or self.batch_size,
            device=self.device,
            workers=2,
            plots=True,
        )
