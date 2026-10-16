"""Collect saved experiment metrics and produce comparison tables/plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def collect_experiments(outputs_dir: str | Path) -> list[dict]:
    rows = []
    for metrics_path in Path(outputs_dir).glob("experiments/**/evaluation/metrics.json"):
        config_path = metrics_path.parents[1] / "config.json"
        if not config_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        row = {
            "dataset": config.get("dataset", {}).get("name", "unknown"),
            "preprocessing": config.get("experiment", {}).get("preprocessing", "unknown"),
            "seed": config.get("experiment", {}).get("seed", "unknown"),
            **{key: metrics.get(key) for key in METRIC_KEYS},
            "inference_seconds": metrics.get("inference_seconds"),
            "sample_count": metrics.get("sample_count"),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["dataset"], row["preprocessing"], str(row["seed"])))


def write_comparison(outputs_dir: str | Path, rows: list[dict]) -> tuple[Path, Path]:
    destination = Path(outputs_dir) / "comparisons"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_path = destination / "macro_f1_comparison.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        axis.plot([row["preprocessing"] for row in subset], [row["macro_f1"] for row in subset], marker="o", label=dataset)
    axis.set(ylabel="Test macro-F1", xlabel="Preprocessing arm", ylim=(0, 1), title="Saved experiment comparison")
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return csv_path, plot_path