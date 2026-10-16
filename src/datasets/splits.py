"""Deterministic, group-aware split creation and integrity checks."""

from __future__ import annotations

import random
from collections import defaultdict

from .adapters import DatasetDescription, ImageRecord


def make_splits(
    description: DatasetDescription,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, tuple[ImageRecord, ...]]:
    if description.official_splits is not None:
        official = {key: tuple(value) for key, value in description.official_splits.items()}
        if "validation" in official:
            return official
        return _split_records(official["train"], seed, ratios, reserve_test=official["test"])
    return _split_records(description.records, seed, ratios)


def _split_records(
    records: tuple[ImageRecord, ...] | list[ImageRecord],
    seed: int,
    ratios: tuple[float, float, float],
    reserve_test: tuple[ImageRecord, ...] = (),
) -> dict[str, tuple[ImageRecord, ...]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be three positive values summing to 1.0.")
    by_class_and_group: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    group_labels: dict[str, str] = {}
    for record in records:
        if record.group in group_labels and group_labels[record.group] != record.label:
            raise ValueError(f"Group {record.group!r} has conflicting labels; group-aware split is unsafe.")
        group_labels[record.group] = record.label
        by_class_and_group[record.label][record.group].append(record)
    rng = random.Random(seed)
    partitions: dict[str, list[ImageRecord]] = {"train": [], "validation": [], "test": list(reserve_test)}
    split_names = ("train", "validation", "test")
    for label in sorted(by_class_and_group):
        groups = list(by_class_and_group[label].items())
        rng.shuffle(groups)
        if reserve_test:
            if len(groups) < 2:
                raise ValueError(f"Class {label!r} needs at least two training groups to form train and validation.")
            train_ratio = ratios[0] / (ratios[0] + ratios[1])
            train_count = min(len(groups) - 1, max(1, round(len(groups) * train_ratio)))
            boundaries = (train_count, len(groups))
        elif len(groups) >= 3:
            train_count = max(1, round(len(groups) * ratios[0]))
            validation_count = max(1, round(len(groups) * ratios[1]))
            if train_count + validation_count >= len(groups):
                train_count = len(groups) - 2
                validation_count = 1
            boundaries = (train_count, train_count + validation_count)
        elif len(groups) == 2:
            boundaries = (1, 1)
        else:
            boundaries = (1, 1)
        for index, (_, group_records) in enumerate(groups):
            split_index = 0 if index < boundaries[0] else (1 if index < boundaries[1] else 2)
            partitions[split_names[split_index]].extend(group_records)
    if not partitions["validation"] or not partitions["test"]:
        raise ValueError("Not enough independent groups to create train/validation/test splits; add data or preserve official splits.")
    result = {name: tuple(sorted(values, key=lambda record: str(record.path))) for name, values in partitions.items()}
    validate_split_integrity(result)
    return result


def validate_split_integrity(splits: dict[str, tuple[ImageRecord, ...]]) -> None:
    paths: dict[str, str] = {}
    groups: dict[str, str] = {}
    for split_name, records in splits.items():
        if not records:
            raise ValueError(f"Split {split_name!r} is empty.")
        for record in records:
            path_key = str(record.path.resolve())
            if path_key in paths and paths[path_key] != split_name:
                raise ValueError(f"Image appears in multiple splits: {record.path}")
            paths[path_key] = split_name
            if record.group in groups and groups[record.group] != split_name:
                raise ValueError(f"Group {record.group!r} leaks across {groups[record.group]} and {split_name}.")
            groups[record.group] = split_name