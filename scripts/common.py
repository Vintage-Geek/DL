"""Shared configuration loading and resolved split reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(config_path)
    return config


def write_split_manifest(path: Path, splits: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "path", "label", "group"))
        writer.writeheader()
        for split_name, records in splits.items():
            for record in records:
                writer.writerow({"split": split_name, "path": str(record.path), "label": record.label, "group": record.group})