"""Train one selected preprocessing arm using validation-only model selection."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch

from scripts.common import load_config, write_split_manifest
from src.datasets import build_dataloaders, inspect_dataset
from src.models import build_resnet50
from src.preprocessing import METHODS
from src.training.reproducibility import seed_everything
from src.training.trainer import train_model


def run(config: dict, method: str) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown preprocessing arm {method!r}; choose from {METHODS}.")
    experiment = config["experiment"]
    seed = int(experiment.get("seed", 42))
    seed_everything(seed, bool(experiment.get("deterministic", True)))
    dataset = config["dataset"]
    description = inspect_dataset(dataset["name"], dataset)
    settings = config["training"]
    data_settings = config["data"]
    loaders, class_to_index, splits = build_dataloaders(
        description,
        seed=seed,
        batch_size=int(settings.get("batch_size", 32)),
        workers=int(settings.get("workers", 0)),
        image_size=int(data_settings.get("image_size", 224)),
        preprocessing_method=method,
        grayscale_compatible=bool(dataset.get("grayscale_compatible", False)),
        ratios=tuple(data_settings.get("split_ratios", (0.7, 0.15, 0.15))),
        imagenet_normalization=bool(data_settings.get("imagenet_normalization", True)),
    )
    device_name = settings.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Set training.device to 'cpu' or install a compatible CUDA build.")
    model = build_resnet50(len(class_to_index), pretrained=bool(settings.get("pretrained", False)))
    output_dir = Path(config.get("outputs_dir", "outputs")) / "experiments" / dataset["name"] / method / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    resolved["experiment"] = {**experiment, "preprocessing": method}
    resolved["runtime"] = {
        "device": str(device),
        "class_names": list(description.classes),
        "class_to_index": class_to_index,
        "class_counts": description.class_counts,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_class_counts": {name: dict(Counter(row.label for row in rows)) for name, rows in splits.items()},
    }
    write_split_manifest(output_dir / "split_manifest.csv", splits)
    print(f"Training {dataset['name']} {method} on {device}; output: {output_dir}")
    train_model(
        model,
        loaders,
        list(description.classes),
        epochs=int(settings.get("epochs", 30)),
        learning_rate=float(settings.get("learning_rate", 1e-4)),
        device=device,
        output_dir=output_dir,
        run_config=resolved,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=METHODS, default=None)
    parser.add_argument("--all-arms", action="store_true", help="Run every method listed in experiment.preprocessing_arms")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if arguments.all_arms:
        arms = config["experiment"].get("preprocessing_arms", ["P0"])
    else:
        arms = [arguments.arm or config["experiment"].get("preprocessing", "P0")]
    for arm in arms:
        run(config, arm)


if __name__ == "__main__":
    main()