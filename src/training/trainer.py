"""Training loop that selects checkpoints using validation macro-F1 only."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

from src.evaluation.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    loaders: dict,
    class_names: list[str],
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: str | Path,
    run_config: dict,
) -> dict:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_count += targets.size(0)

        validation_targets, validation_predictions = _predict(model, loaders["validation"], device)
        validation_metrics = compute_metrics(validation_targets, validation_predictions, class_names)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_row)
        LOGGER.info("Epoch %d/%d: %s", epoch, epochs, epoch_row)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "run_config": run_config,
            "epoch": epoch,
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        torch.save(checkpoint, output / "last.pt")
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(checkpoint, output / "best.pt")

    _write_csv(output / "training_log.csv", history)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2, default=str)
    result = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "training_seconds": time.perf_counter() - started,
        "epochs_completed": epochs,
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def _predict(model: nn.Module, loader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    targets_all, predictions_all = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            targets_all.extend(targets.tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    return targets_all, predictions_all


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)