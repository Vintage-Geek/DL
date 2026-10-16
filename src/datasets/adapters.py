"""Explicit dataset discovery for class-folder datasets and HAM10000 metadata."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OFFICIAL_SPLITS = {
    "train": ("train", "training"),
    "validation": ("val", "validation", "valid"),
    "test": ("test", "testing"),
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    group: str


@dataclass(frozen=True)
class DatasetDescription:
    name: str
    records: tuple[ImageRecord, ...]
    classes: tuple[str, ...]
    class_counts: dict[str, int]
    official_splits: dict[str, tuple[ImageRecord, ...]] | None = None


def _images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _records_from_class_folders(directory: Path) -> list[ImageRecord]:
    class_directories = sorted(path for path in directory.iterdir() if path.is_dir())
    if len(class_directories) < 2:
        raise ValueError(f"Expected at least two class folders in {directory}; found {len(class_directories)}.")
    records = []
    for class_directory in class_directories:
        image_paths = _images(class_directory)
        if not image_paths:
            raise ValueError(f"Class folder {class_directory} contains no supported images.")
        records.extend(ImageRecord(path, class_directory.name, str(path.resolve())) for path in image_paths)
    return records


def _find_official_splits(root: Path) -> dict[str, tuple[ImageRecord, ...]] | None:
    found: dict[str, Path] = {}
    directories = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    for split, candidates in OFFICIAL_SPLITS.items():
        for candidate in candidates:
            directory = directories.get(candidate)
            if directory is not None:
                found[split] = directory
                break
    if found and not {"train", "test"}.issubset(found):
        raise ValueError(
            f"Found official split directories {sorted(found)}, but both train/training and test/testing are required. "
            "Point dataset.root to the class-folder root or provide the complete official split layout."
        )
    if "train" not in found or "test" not in found:
        return None
    result = {split: tuple(_records_from_class_folders(directory)) for split, directory in found.items()}
    train_classes = {record.label for record in result["train"]}
    for split, records in result.items():
        split_classes = {record.label for record in records}
        if split_classes != train_classes:
            missing = sorted(train_classes - split_classes)
            extra = sorted(split_classes - train_classes)
            raise ValueError(f"Official {split} class folders differ from training; missing={missing}, extra={extra}.")
    return result


def _read_ham10000(root: Path, settings: dict) -> list[ImageRecord]:
    metadata_value = settings.get("metadata_csv")
    if not metadata_value:
        raise ValueError("HAM10000 requires dataset.metadata_csv pointing to the supplied metadata CSV.")
    metadata_path = Path(metadata_value)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {metadata_path}")
    columns = settings.get("columns", {})
    image_column = columns.get("image_id", "image_id")
    label_column = columns.get("label", "dx")
    lesion_column = columns.get("lesion_id", "lesion_id")
    configured_image_dirs = settings.get("images_dirs", [settings.get("images_dir", ".")])
    images_dirs = [Path(value) if Path(value).is_absolute() else root / value for value in configured_image_dirs]
    records = []
    missing = []
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        required = {image_column, label_column}
        absent = required - headers
        if absent:
            raise ValueError(f"HAM10000 metadata is missing required columns: {sorted(absent)}.")
        for row in reader:
            image_id = row[image_column].strip()
            label = row[label_column].strip()
            if not image_id or not label:
                continue
            candidate = Path(image_id)
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                candidates = [directory / f"{image_id}{extension}" for directory in images_dirs for extension in (".jpg", ".png", ".jpeg")]
            else:
                candidates = [directory / candidate for directory in images_dirs]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                missing.append(image_id)
                continue
            lesion_id = row.get(lesion_column, "").strip() if lesion_column in headers else ""
            records.append(ImageRecord(image_path, label, lesion_id or image_id))
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} metadata images were not found under {images_dirs} (e.g. {examples}).")
    if not records:
        raise ValueError("HAM10000 metadata produced no image records; check column mappings and files.")
    return records


def inspect_dataset(name: str, settings: dict) -> DatasetDescription:
    """Validate configured data and return records without downloading or guessing paths."""
    root_value = settings.get("root")
    if not root_value:
        raise ValueError("dataset.root must be configured to a local dataset directory.")
    root = Path(root_value).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}. See docs/DATASETS.md for setup.")
    adapter = settings.get("adapter", "imagefolder")
    official_splits = None
    if adapter == "ham10000":
        records = _read_ham10000(root, settings)
    elif adapter in {"imagefolder", "brain_tumor_mri", "retinal_oct", "colorectal_he"}:
        official_splits = _find_official_splits(root)
        records = [record for split in official_splits.values() for record in split] if official_splits else _records_from_class_folders(root)
    else:
        raise ValueError(f"Unsupported dataset adapter {adapter!r}; use imagefolder or ham10000.")
    classes = tuple(sorted({record.label for record in records}))
    counts = dict(Counter(record.label for record in records))
    return DatasetDescription(name, tuple(records), classes, counts, official_splits)