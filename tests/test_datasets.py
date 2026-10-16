import csv

import pytest
from PIL import Image

from src.datasets.adapters import ImageRecord, inspect_dataset
from src.datasets.splits import make_splits, validate_split_integrity


def _write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_class_folder_discovery_and_deterministic_group_splits(tmp_path):
    for label in ("healthy", "disease"):
        for index in range(10):
            _write_image(tmp_path / label / f"{index}.png", (index * 10, 30, 60))
    description = inspect_dataset("synthetic", {"root": str(tmp_path)})
    assert description.classes == ("disease", "healthy")
    assert description.class_counts == {"disease": 10, "healthy": 10}
    first = make_splits(description, seed=17)
    second = make_splits(description, seed=17)
    assert {key: [row.path for row in value] for key, value in first.items()} == {
        key: [row.path for row in value] for key, value in second.items()
    }
    validate_split_integrity(first)


def test_missing_dataset_root_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        inspect_dataset("missing", {"root": str(tmp_path / "absent")})


def test_ham_metadata_uses_lesion_as_group(tmp_path):
    images = tmp_path / "images"
    for index in range(7):
        _write_image(images / f"image-{index}.jpg", (index * 10, 20, 30))
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "dx", "lesion_id"])
        writer.writeheader()
        writer.writerows([
            {"image_id": "image-0", "dx": "a", "lesion_id": "lesion-1"},
            {"image_id": "image-1", "dx": "a", "lesion_id": "lesion-1"},
            {"image_id": "image-2", "dx": "a", "lesion_id": "lesion-2"},
            {"image_id": "image-3", "dx": "a", "lesion_id": "lesion-3"},
            {"image_id": "image-4", "dx": "b", "lesion_id": "lesion-4"},
            {"image_id": "image-5", "dx": "b", "lesion_id": "lesion-5"},
            {"image_id": "image-6", "dx": "b", "lesion_id": "lesion-6"},
        ])
    description = inspect_dataset("ham", {
        "root": str(tmp_path), "adapter": "ham10000", "metadata_csv": "metadata.csv", "images_dir": "images"
    })
    splits = make_splits(description, seed=4)
    validate_split_integrity(splits)
    assignments = {
        record.group: split_name
        for split_name, records in splits.items()
        for record in records
    }
    assert assignments["lesion-1"] in {"train", "validation", "test"}
    assert len(assignments) == 6
    for split_name in ("train", "validation", "test"):
        assert {record.label for record in splits[split_name]} == {"a", "b"}


def test_capitalized_official_train_test_folders_are_discovered(tmp_path):
    for split in ("Training", "Testing"):
        for label in ("class-a", "class-b"):
            _write_image(tmp_path / split / label / "one.png", (10, 20, 30))
            if split == "Training":
                _write_image(tmp_path / split / label / "two.png", (20, 30, 40))
    description = inspect_dataset("official", {"root": str(tmp_path)})
    assert description.official_splits is not None
    splits = make_splits(description)
    assert {record.path.parts[-3] for record in splits["test"]} == {"Testing"}
    assert {record.path.parts[-3] for record in splits["train"]} == {"Training"}


def test_incomplete_or_inconsistent_official_splits_fail(tmp_path):
    for label in ("class-a", "class-b"):
        _write_image(tmp_path / "train" / label / "one.png", (10, 20, 30))
    _write_image(tmp_path / "test" / "class-a" / "one.png", (10, 20, 30))
    _write_image(tmp_path / "test" / "class-c" / "one.png", (30, 40, 50))
    with pytest.raises(ValueError, match="class folders differ"):
        inspect_dataset("inconsistent", {"root": str(tmp_path)})

    incomplete = tmp_path / "incomplete"
    for label in ("class-a", "class-b"):
        _write_image(incomplete / "train" / label / "one.png", (10, 20, 30))
    with pytest.raises(ValueError, match="both train/training and test/testing"):
        inspect_dataset("incomplete", {"root": str(incomplete)})


def test_group_split_rejects_conflicting_labels(tmp_path):
    rows = [
        ImageRecord(tmp_path / "a.png", "a", "same"),
        ImageRecord(tmp_path / "b.png", "b", "same"),
    ]
    from src.datasets.adapters import DatasetDescription

    with pytest.raises(ValueError, match="conflicting labels"):
        make_splits(DatasetDescription("bad", tuple(rows), ("a", "b"), {"a": 1, "b": 1}))