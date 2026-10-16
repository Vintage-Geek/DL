"""Sklearn-backed metrics, timing, and confusion matrix artifacts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_metrics(targets: list[int], predictions: list[int], class_names: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if not targets or len(targets) != len(predictions):
        raise ValueError("targets and predictions must be non-empty and have equal length.")
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "sample_count": len(targets),
    }


def evaluate_model(model, loader, class_names: list[str], device: torch.device, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    targets_all, predictions_all = [], []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())
    elapsed = time.perf_counter() - inference_started
    metrics = compute_metrics(targets_all, predictions_all, class_names)
    metrics["inference_seconds"] = elapsed
    metrics["inference_seconds_per_image"] = elapsed / max(len(targets_all), 1)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("class", "precision", "recall", "f1", "support"))
        writer.writeheader()
        for name, values in metrics["per_class"].items():
            writer.writerow({"class": name, **values})
    matrix = np.asarray(metrics["confusion_matrix"])
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    figure, axis = plt.subplots(figsize=(max(5, len(class_names) * 0.7), max(4, len(class_names) * 0.6)))
    axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="Actual")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics