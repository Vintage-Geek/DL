# Dataset Setup and Adapter Contract

All datasets are local-only. Download/access them through their official sources, honor access restrictions, and do not commit data. The code does not infer labels from filenames, download archives, or claim a particular archive was inspected. Set `dataset.root` to the extracted local root in the experiment JSON. Class labels are discovered from the actual class-directory names or configured metadata values.

## Supported Sources

| Dataset | Source | Adapter and expected local data |
| --- | --- | --- |
| Brain Tumor MRI | [Kaggle dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | `brain_tumor_mri`; class folders directly under root, or class folders under official `train`/`training`, optional `val`/`validation`/`valid`, and `test`/`testing` directories. Split names are case-insensitive. |
| Retinal OCT | [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3) | `retinal_oct`; the configured root must directly contain class folders, or recognized `train`/`training`, optional validation, and `test`/`testing` directories whose immediate children are class folders. The downloaded archive layout has not been inspected by this project. |
| Colorectal H&E | [Zenodo record 1214456](https://zenodo.org/records/1214456) | `colorectal_he`; class folders directly under root, optionally inside recognized official splits. Flat images need a documented label mapping and class-folder organization before this adapter can validate them. |
| HAM10000 | [Harvard Dataverse record](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | `ham10000`; configured metadata CSV plus image files discoverable under configured image directory/directories. Defaults are `image_id`, `dx`, and `lesion_id`; map columns to the actual supplied metadata. |

The first three adapters use the discovered folder names as class labels; they require at least two class folders and at least one supported image per class. Recognized images are JPG/JPEG, PNG, BMP, TIF/TIFF. When an official train and test directory pair is detected, each must contain the training classes; optional official validation is retained. If validation is absent, only official training records are subdivided. Other layouts fail with a message; there is no fuzzy recursive search for a plausible dataset root.

Example class-folder layout:

```text
data/your_dataset/
  train/                 # optional official split
    class_name_1/*.jpg
    class_name_2/*.png
  validation/            # optional official split
    class_name_1/*.jpg
    class_name_2/*.jpg
  test/                  # optional official split
    class_name_1/*.jpg
    class_name_2/*.jpg
```

For Retinal OCT, set `dataset.root` to the directory immediately above the class folders (generated splits) or immediately above the split directories (directory splits). An accepted generated-split layout is:

```text
data/retinal_oct_pilot/
  class_name_1/*.jpg
  class_name_2/*.jpg
  class_name_3/*.jpg
```

An accepted recognized split-directory layout is:

```text
data/retinal_oct/
  train/                 # alternatively training/
    class_name_1/*.jpg
    class_name_2/*.jpg
  validation/            # optional; alternatively val/ or valid/
    class_name_1/*.jpg
    class_name_2/*.jpg
  test/                   # alternatively testing/
    class_name_1/*.jpg
    class_name_2/*.jpg
```

Labels are the immediate class-directory names, exactly as present on disk; this project does not impose a retinal class-name list or infer labels from filenames. The recognized image formats are JPG/JPEG, PNG, BMP, and TIF/TIFF. If the extracted archive has a wrapper directory, nested class folders, a flat image collection, or a different split convention, do not assume it is accepted: inspect the archive first, then set `root` to the directory matching this contract or organize a documented local copy. The validator reports recordable counts and structural problems but does not authenticate the archive against the source.

The starter pilot config is [configs/retinal_oct.pilot.example.json](../configs/retinal_oct.pilot.example.json). Its root is intentionally `data/retinal_oct_pilot`; make a small local subset in the generated-split layout above before using it. Use at least three images per class for a generated train/validation/test split. This is only a small plumbing pilot: the current folder adapter treats each image path as its group and cannot establish patient-level independence for OCT.

## Validator Report

Run `python -m scripts.validate_dataset --config <config.json>` from the repository root. The JSON report includes class and split counts, class counts within each split, supported extensions, corrupt supported-format images (decoded with OpenCV), recognized unsupported image formats (`.gif`, `.webp`, `.dcm`, `.jp2`, `.j2k`, `.ppm`, `.pgm`, `.pnm`), supported images that were not assigned a discovered label, blank HAM10000 image-ID/label rows, and up to 20 group-identifier examples. A nonzero exit status means a discovery error or one of these data issues was found. The unsupported-format check is limited to the listed recognizable extensions; it is not a general-purpose file-format detector.

Split provenance is reported conservatively. `automatically_generated_seeded_group_split` means the adapter found class folders and created partitions. `recognized_official_split_directory_names` means folder names matched the project's train/validation/test aliases; `official_source_authenticated` remains false because directory names alone do not prove a source-provided official split. If validation is missing, the report identifies validation as generated from the training directory. Verify any official split claims against the dataset documentation and retain source/version evidence yourself.

For a full local extraction, copy the pilot example to a local config, set `dataset.root` to the extracted directory that matches one of the layouts above, and validate that config before training. Keep a separate pilot subset root in the pilot config so a pilot run does not silently train across the full dataset.

Or put class folders immediately inside the configured root to request deterministic generated splits. Do not mix a split layout with class directories at the root. For a flat archive, organize with directory links/copies according to the authoritative labels; this project intentionally does not infer labels from file naming patterns.

## HAM10000 Metadata

Configure the CSV relative to `dataset.root` (or with an absolute path), along with image paths and metadata column names:

```json
{
  "name": "ham10000",
  "adapter": "ham10000",
  "root": "data/ham10000",
  "metadata_csv": "HAM10000_metadata.csv",
  "images_dirs": ["HAM10000_images_part_1", "HAM10000_images_part_2"],
  "columns": {
    "image_id": "image_id",
    "label": "dx",
    "lesion_id": "lesion_id"
  },
  "grayscale_compatible": false
}
```

Image identifiers can include an extension, or the adapter tries `.jpg`, `.png`, and `.jpeg`. Relative image directories are under `root`. Missing required columns, absent image files, empty metadata, and missing roots are errors with examples. If lesion ID is absent or blank, that row's image ID is its group; if the dataset provides a different patient/lesion grouping column, set `columns.lesion_id` to it. A group with inconsistent class labels is rejected. The dataset does not provide a configured official split in this project, so the configured seed and group-aware split are used.

## Splits and Limitations

Generated splits use class-wise groups and the configured ratios (default 70/15/15). At least three independent groups per class are needed to put that class into all three generated partitions; with fewer groups the resulting test support can be absent for a class or the global split can fail. For official train/test layouts without validation, at least two groups in each training class are required to produce train and validation. Rare classes should be reviewed in `split_manifest.csv` and per-class supports; do not silently report unsupported test classes as evidence of performance. HAM10000 grouping prevents lesion overlap only when its metadata group column is correctly mapped. No patient-level grouping beyond configured metadata is inferred.

The validation command reports discovered classes, class counts and split counts and decodes every discovered supported image to detect corruption. Keep a note of source version, archive checksums, any filesystem reorganization, and metadata mapping as part of the final study record.