# A Comparative Analysis of Image Preprocessing Techniques for Explainable Medical Image Classification

A reproducible, local research pipeline comparing image preprocessing arms with one fixed classifier: **ResNet-50**. It supports brain tumor MRI, retinal OCT, colorectal H&E, and HAM10000 through configurable local roots. It never downloads datasets, trains from the demo, or claims experimental results that have not been produced locally.

## Implemented

- ResNet-50 classification with optional, explicitly enabled ImageNet-pretrained weights. By default, model construction does not download weights.
- Dataset validation and class-folder adapters for MRI, OCT, and colorectal H&E, plus a CSV/lesion-aware HAM10000 adapter.
- Deterministic class-stratified, group-aware train/validation/test splits; detected official train/test layouts are retained, with validation carved only from official training data when needed.
- P0-P9 and CLAHE image preprocessing, fixed-size resizing, ImageNet normalization, seeded training, validation-only best-checkpoint selection, logs, checkpoints, and test reports.
- Accuracy, macro precision/recall/F1, per-class metrics, support counts, confusion matrices, inference timing, comparison CSV/plot, and Grad-CAM overlays.
- Streamlit inference against saved checkpoints, with class probabilities and local prediction artifacts.
- Synthetic, offline pytest coverage. Tests do not download datasets or weights.

Not implemented: quantitative explanation stability, repeated-seed statistical aggregation, pretrained-weight retrieval, or any claim of clinical validity. Grad-CAM is a qualitative visualization, not evidence that a model is clinically valid or relying on causal features.

## Environment

The package declares Python `>=3.11,<3.14`; Python 3.12 is the recommended clean setup. In Windows PowerShell, from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python --version
python -m pip check
```

Confirm the version output is Python 3.12. `pyproject.toml` reads the exact dependency pins from `requirements.txt`, so `python -m pip install -e .` installs the project and those same pins in one step. No package upgrades or CUDA-specific replacements are part of the default setup. If the Python launcher is unavailable, use an installed Python 3.12 executable explicitly, for example `C:\Path\To\Python312\python.exe -m venv .venv`. If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -e .` and use `.venv\Scripts\python.exe` for commands.

The pinned PyTorch wheel runs on CPU without a CUDA setup. For NVIDIA GPUs, first install the matching PyTorch/torchvision CUDA wheels from the official PyTorch selector for your driver/CUDA combination, then install the remaining dependencies from `requirements.txt` (omit or replace its two PyTorch lines to avoid replacing those wheels). Set `training.device` to `"auto"`, `"cpu"`, or an available device such as `"cuda"`. CUDA is optional.

## Dataset Setup

Obtain data yourself from the source links and comply with each source's access conditions and license. No images or credentials belong in Git. Detailed layouts, metadata columns, split behavior, and adapter constraints are in [docs/DATASETS.md](docs/DATASETS.md).

The experiment file is ordinary JSON. Copy [configs/experiment.example.json](configs/experiment.example.json) to a working config such as `configs/brain_mri.json`; set `dataset.root` to the local dataset root. For HAM10000, also set `metadata_csv`, `images_dir` or `images_dirs`, and actual metadata column names. Paths are resolved relative to the current working directory. The Retinal OCT layout accepted by this repository and its pilot configuration are documented in [docs/DATASETS.md](docs/DATASETS.md) and [configs/retinal_oct.pilot.example.json](configs/retinal_oct.pilot.example.json). The adapter does not inspect or reorganize the downloaded archive automatically.

## Commands

Run from the repository root with the activated environment:

```powershell
# Validate the configured local data and deterministic split
python -m scripts.validate_dataset --config configs/brain_mri.json

# Validate a locally configured Retinal OCT dataset before training
Copy-Item configs\retinal_oct.pilot.example.json configs\retinal_oct.local.json
# Edit dataset.root in retinal_oct.local.json to the extracted dataset root
python -m scripts.validate_dataset --config configs/retinal_oct.local.json

# Run offline tests (no real data required)
python -m pytest

# Train one arm; epochs, batch size, learning rate, workers, device and seed live in JSON
python -m scripts.train --config configs/brain_mri.json --arm P0

# Evaluate that run's validation-selected checkpoint on the held-out test split
python -m scripts.evaluate --run-dir outputs/experiments/brain_tumor_mri/P0/seed-42

# One-epoch ResNet-50 P0 pilot on the small subset at dataset.root
python -m scripts.train --config configs/retinal_oct.pilot.example.json --arm P0

# Evaluate that pilot checkpoint
python -m scripts.evaluate --run-dir outputs/experiments/retinal_oct/P0/seed-42

# Train/evaluate each configured arm sequentially, then write saved-results comparison
python -m scripts.run_experiments --config configs/brain_mri.json

# Or compare only experiments that already have evaluation reports
python -m scripts.compare

# Generate a local Grad-CAM overlay and prediction record for one image
python -m scripts.explain --run-dir outputs/experiments/brain_tumor_mri/P0/seed-42 --image path/to/image.jpg

# Launch the inference-only demo
python -m streamlit run app.py
```

The example uses 30 epochs, not a one-epoch placeholder. Those defaults are a starting protocol, not a claim that a run is adequate; choose and document the protocol before experiments. Training is an explicit CLI job. It saves `best.pt` (chosen by validation macro-F1), `last.pt`, `config.json`, `training_log.csv`, `training_summary.json`, and `split_manifest.csv`. Evaluation writes `metrics.json`, `per_class_metrics.csv`, and confusion matrices under `evaluation/`. The demo reads saved checkpoints and never trains on uploads.

Use validation metrics to choose preprocessing/training settings. The test set is held back for final reporting only; do not select an arm, tune parameters, or make repeated decisions based on test metrics. `scripts.compare` is a reporting utility, not a model-selection routine. Repeat runs with distinct configured seeds for variability; the supplied comparison keeps seed-level rows and does not calculate statistical intervals.

## Preprocessing Protocol

All images are decoded as RGB, resized to the same configured size (default 224 x 224), and receive ImageNet channel normalization by default. Each arm then applies exactly one configured transformation before that normalization: P0 raw pixels; P1 fixed 1.15 brightness gain with clipping; P2 1st-to-99th percentile contrast stretch; P3 grayscale then contrast stretch (only when explicitly enabled for a grayscale-compatible dataset, with RGB restored for ResNet); P4 global histogram equalization (luminance for RGB); P5 3 x 3 mean; P6 3 x 3 median; P7 Gaussian smoothing sigma 1.0; P8 per-image, per-channel min-max; P9 per-image, per-channel z-score; CLAHE (luminance for RGB). P9's genuine z-scores are subsequently passed through the configured ImageNet normalization just like other arms.

P3 is deliberately opt-in: do not enable it for color H&E or dermoscopy unless the scientific protocol explicitly justifies loss of color information. MRI and OCT are often grayscale, but the pipeline reads them as RGB so every arm has a consistent three-channel ResNet input. `configs/experiment.example.json` includes P3 because its example dataset is configured as grayscale-compatible; remove P3 from the arm list for RGB datasets.

## Reproducibility and Outputs

The seed covers Python, NumPy and PyTorch; deterministic cuDNN behavior is requested by default. Splits are regenerated deterministically from the same seed and written as a manifest in every run. HAM10000 uses the configured lesion ID column as a grouping key when available, falling back to image ID; related lesions are kept in one split. Official class-folder train/validation/test directories take precedence when detected. If only official train/test are present, the official test records remain untouched and validation is derived from official training records.

`outputs/` and `data/` are ignored by Git (except `outputs/.gitkeep`). Runs record per-class supports and split counts to expose imbalance. Single-seed metrics are not estimates of between-seed variability. Inference time is a local rough measurement and depends on hardware, workers, and batch size.