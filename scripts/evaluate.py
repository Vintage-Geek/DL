"""Evaluate a saved best checkpoint once on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.common import load_config
from src.datasets import build_dataloaders, inspect_dataset
from src.evaluation import evaluate_model
from src.models import load_checkpoint


def evaluate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    dataset = config["dataset"]
    description = inspect_dataset(dataset["name"], dataset)
    training = config["training"]
    data = config["data"]
    method = config["experiment"]["preprocessing"]
    loaders, _, _ = build_dataloaders(
        description,
        seed=int(config["experiment"].get("seed", 42)),
        batch_size=int(training.get("batch_size", 32)),
        workers=int(training.get("workers", 0)),
        image_size=int(data.get("image_size", 224)),
        preprocessing_method=method,
        grayscale_compatible=bool(dataset.get("grayscale_compatible", False)),
        ratios=tuple(data.get("split_ratios", (0.7, 0.15, 0.15))),
        imagenet_normalization=bool(data.get("imagenet_normalization", True)),
    )
    if "test" not in loaders:
        raise ValueError("No test split was produced; evaluation is not permitted without a held-out test split.")
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device=device)
    if checkpoint["class_names"] != list(description.classes):
        raise ValueError("Checkpoint classes differ from discovered dataset classes; refusing incorrect evaluation.")
    metrics = evaluate_model(model, loaders["test"], checkpoint["class_names"], device, run_dir / "evaluation")
    print(json.dumps({key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "sample_count", "inference_seconds")}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Experiment directory containing config.json and best.pt")
    arguments = parser.parse_args()
    evaluate_run(arguments.run_dir)


if __name__ == "__main__":
    main()