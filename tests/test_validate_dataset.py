import csv

from PIL import Image

from scripts.validate_dataset import build_validation_report


def _write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_validator_reports_corrupt_unsupported_and_unlabelled_images(tmp_path):
    root = tmp_path / "images"
    for label in ("normal", "disease"):
        for index in range(3):
            _write_image(root / label / f"{index}.png", (index * 20, 40, 60))
    (root / "normal" / "broken.png").write_bytes(b"not an image")
    (root / "normal" / "unsupported.webp").write_bytes(b"webp placeholder")
    _write_image(root / "unlabelled.png", (0, 0, 0))
    report = build_validation_report({
        "dataset": {"name": "retinal_oct", "adapter": "retinal_oct", "root": str(root)},
        "experiment": {"seed": 7},
        "data": {"image_size": 16},
    })

    assert report["class_counts"] == {"disease": 3, "normal": 4}
    assert report["split_provenance"]["strategy"] == "automatically_generated_seeded_group_split"
    assert report["corrupt_images"]["count"] == 1
    assert report["unsupported_images"]["count"] == 1
    assert report["images_without_labels"]["count"] == 1
    assert report["group_identifiers"]["source"].startswith("image path")
    assert not report["valid"]


def test_validator_reports_blank_ham_labels_and_detected_split_provenance(tmp_path):
    root = tmp_path / "ham"
    image_dir = root / "images"
    metadata_path = root / "metadata.csv"
    rows = []
    for label in ("a", "b"):
        for index in range(3):
            image_id = f"{label}-{index}"
            _write_image(image_dir / f"{image_id}.jpg", (index * 20, 40, 60))
            rows.append({"image_id": image_id, "dx": label, "lesion_id": f"lesion-{image_id}"})
    rows.append({"image_id": "blank-label", "dx": "", "lesion_id": "lesion-blank"})
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "dx", "lesion_id"))
        writer.writeheader()
        writer.writerows(rows)

    report = build_validation_report({
        "dataset": {
            "name": "ham10000",
            "adapter": "ham10000",
            "root": str(root),
            "metadata_csv": "metadata.csv",
            "images_dir": "images",
        },
        "experiment": {"seed": 7},
        "data": {"image_size": 16},
    })

    assert report["metadata_rows_missing_label"] == 1
    assert report["group_identifiers"]["unique_count"] == 6
    assert report["group_identifiers"]["source"] == "lesion_id"
    assert "automatically_generated" in report["split_provenance"]["strategy"]
    assert not report["valid"]


def test_validator_does_not_authenticate_split_names_as_official(tmp_path):
    root = tmp_path / "retinal"
    for split in ("train", "test"):
        for label in ("normal", "disease"):
            image_count = 2 if split == "train" else 1
            for index in range(image_count):
                _write_image(root / split / label / f"{index}.png", (index * 20, 40, 60))

    report = build_validation_report({
        "dataset": {"name": "retinal_oct", "adapter": "retinal_oct", "root": str(root)},
        "experiment": {"seed": 7},
        "data": {"image_size": 16},
    })

    assert report["split_provenance"]["strategy"] == "recognized_official_split_directory_names"
    assert report["split_provenance"]["official_directory_names_detected"]
    assert not report["split_provenance"]["official_source_authenticated"]
    assert report["split_provenance"]["validation_source"] == "generated_from_training_directory"
    assert report["valid"]