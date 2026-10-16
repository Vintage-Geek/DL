"""Validate local data roots and print sample/class/split counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2

from src.datasets.adapters import IMAGE_EXTENSIONS
from src.datasets import build_dataloaders, inspect_dataset
from scripts.common import load_config

KNOWN_UNSUPPORTED_IMAGE_EXTENSIONS = {".gif", ".webp", ".dcm", ".jp2", ".j2k", ".ppm", ".pgm", ".pnm"}
MAX_EXAMPLES = 20


def _metadata_label_issues(dataset: dict) -> dict:
    if dataset.get("adapter") != "ham10000":
        return {"rows_missing_image_id": 0, "rows_missing_label": 0, "examples": []}
    root = Path(dataset["root"]).expanduser()
    metadata_path = Path(dataset["metadata_csv"])
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    columns = dataset.get("columns", {})
    image_column = columns.get("image_id", "image_id")
    label_column = columns.get("label", "dx")
    missing_image_ids = []
    missing_labels = []
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing_columns = {image_column, label_column} - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"HAM10000 metadata is missing configured columns: {sorted(missing_columns)}.")
        for row_number, row in enumerate(reader, start=2):
            image_id = (row.get(image_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not image_id:
                missing_image_ids.append(row_number)
            if not label:
                missing_labels.append(row_number)
    return {
        "rows_missing_image_id": len(missing_image_ids),
        "rows_missing_label": len(missing_labels),
        "examples": [f"CSV row {row}" for row in sorted(set(missing_image_ids + missing_labels))[:MAX_EXAMPLES]],
    }


def build_validation_report(config: dict) -> dict:
    dataset = config["dataset"]
    root = Path(dataset.get("root", ".")).expanduser()
    issues = []
    report = {
        "dataset": dataset.get("name", "unknown"),
        "root": str(root),
        "supported_image_extensions": sorted(IMAGE_EXTENSIONS),
        "class_counts": {},
        "split_counts": {},
        "split_class_counts": {},
        "split_provenance": {},
        "corrupt_images": {"count": 0, "examples": []},
        "unsupported_images": {"count": 0, "examples": []},
        "images_without_labels": {"count": 0, "examples": []},
        "metadata_rows_missing_image_id": 0,
        "metadata_rows_missing_label": 0,
        "group_identifiers": {},
        "issues": issues,
    }

    try:
        metadata_issues = _metadata_label_issues(dataset)
        report["metadata_rows_missing_image_id"] = metadata_issues["rows_missing_image_id"]
        report["metadata_rows_missing_label"] = metadata_issues["rows_missing_label"]
        if metadata_issues["rows_missing_image_id"] or metadata_issues["rows_missing_label"]:
            issues.append("Metadata contains rows with missing image identifiers or labels.")

        description = inspect_dataset(dataset.get("name", "unknown"), dataset)
        report["class_counts"] = description.class_counts
        loaders, _, splits = build_dataloaders(
            description,
            seed=int(config.get("experiment", {}).get("seed", 42)),
            batch_size=1,
            workers=0,
            image_size=int(config.get("data", {}).get("image_size", 224)),
            preprocessing_method="P0",
            grayscale_compatible=bool(dataset.get("grayscale_compatible", False)),
            ratios=tuple(config.get("data", {}).get("split_ratios", (0.7, 0.15, 0.15))),
            imagenet_normalization=bool(config.get("data", {}).get("imagenet_normalization", True)),
        )
        del loaders
        report["split_counts"] = {name: len(records) for name, records in splits.items()}
        report["split_class_counts"] = {
            name: dict(Counter(record.label for record in records))
            for name, records in splits.items()
        }
        if description.official_splits is None:
            report["split_provenance"] = {
                "strategy": "automatically_generated_seeded_group_split",
                "official_directory_names_detected": False,
                "official_source_authenticated": False,
            }
        else:
            report["split_provenance"] = {
                "strategy": "recognized_official_split_directory_names",
                "official_directory_names_detected": True,
                "official_source_authenticated": False,
                "validation_source": (
                    "recognized_validation_directory"
                    if "validation" in description.official_splits
                    else "generated_from_training_directory"
                ),
            }

        record_paths = {record.path.resolve() for record in description.records}
        all_files = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
        corrupt = [
            record.path
            for record in description.records
            if cv2.imread(str(record.path), cv2.IMREAD_COLOR) is None
        ]
        unsupported = [
            path for path in all_files
            if path.suffix.lower() in KNOWN_UNSUPPORTED_IMAGE_EXTENSIONS
        ]
        unassigned = [
            path for path in all_files
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() not in record_paths
        ]
        for key, paths in (("corrupt_images", corrupt), ("unsupported_images", unsupported), ("images_without_labels", unassigned)):
            report[key] = {"count": len(paths), "examples": [str(path) for path in paths[:MAX_EXAMPLES]]}
            if paths:
                issues.append(f"Found {len(paths)} {key.replace('_', ' ')}.")

        group_counts = Counter(record.group for record in description.records)
        group_values = sorted(group_counts)
        report["group_identifiers"] = {
            "source": (
                dataset.get("columns", {}).get("lesion_id", "lesion_id")
                if dataset.get("adapter") == "ham10000"
                else "image path; no patient/subject identifier configured"
            ),
            "blank_ham_group_fallback": "image_id" if dataset.get("adapter") == "ham10000" else None,
            "unique_count": len(group_values),
            "examples": group_values[:MAX_EXAMPLES],
            "examples_truncated": len(group_values) > MAX_EXAMPLES,
        }
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        report["discovery_error"] = str(error)
        issues.append(str(error))

    report["valid"] = not issues
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment JSON")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    report = build_validation_report(config)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()