"""Local Streamlit demo for saved ResNet-50 experiments."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.explainability import GradCAM, save_gradcam_overlay
from src.models import load_checkpoint
from src.preprocessing import METHODS, PreprocessingPipeline


st.set_page_config(page_title="Medical Image Research Demo", layout="wide")
st.title("Medical Image Classification")
st.caption("Research prototype. Predictions and Grad-CAM visualizations are not clinical advice.")


@st.cache_resource(show_spinner="Loading ResNet-50 checkpoint...")
def load_model_once(checkpoint_path: str, device_name: str):
    return load_checkpoint(checkpoint_path, device=device_name)


def available_runs(dataset_name: str) -> dict[str, tuple[Path, dict]]:
    choices = {}
    for checkpoint_path in Path("outputs/experiments").glob("**/best.pt"):
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("dataset", {}).get("name") == dataset_name:
            arm = config.get("experiment", {}).get("preprocessing", "unknown")
            choices[arm] = (checkpoint_path, config)
    return choices


registry_path = Path("configs/datasets.json")
if not registry_path.is_file():
    st.error("Dataset registry is missing: configs/datasets.json")
    st.stop()
with registry_path.open(encoding="utf-8") as stream:
    registry = json.load(stream)

dataset_id = st.selectbox("Dataset", list(registry), format_func=lambda key: registry[key]["name"])
runs = available_runs(dataset_id)
if not runs:
    st.info("No saved model runs found for this dataset. Train and evaluate an experiment first.")
    st.stop()
method = st.selectbox("Preprocessing model", sorted(runs))
checkpoint_path, run_config = runs[method]
uploaded = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png", "bmp", "tif", "tiff"))

if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    original_rgb = np.asarray(original)
    columns = st.columns(2)
    columns[0].image(original, caption="Input image", use_container_width=True)
    infer_button = st.button("Predict and explain", type="primary")
    if infer_button:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        model, checkpoint = load_model_once(str(checkpoint_path.resolve()), device_name)
        image_size = int(run_config.get("data", {}).get("image_size", 224))
        resized = cv2.resize(original_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        dataset_config = run_config.get("dataset", {})
        processed = PreprocessingPipeline(method, bool(dataset_config.get("grayscale_compatible", False)))(resized)
        input_tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
        if run_config.get("data", {}).get("imagenet_normalization", True):
            mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            standard_deviation = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            input_tensor = (input_tensor - mean) / standard_deviation
        input_tensor = input_tensor.unsqueeze(0).to(device_name)
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        predicted_index = int(probabilities.argmax())
        explainer = GradCAM(model)
        heatmap, _ = explainer(input_tensor)
        explainer.close()

        prediction_id = uuid.uuid4().hex[:12]
        prediction_dir = Path("outputs/predictions") / prediction_id
        prediction_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original_rgb).save(prediction_dir / "original.png")
        save_gradcam_overlay(original_rgb, heatmap, prediction_dir / "gradcam_overlay.png")
        prediction_record = {
            "dataset": dataset_id,
            "preprocessing": method,
            "checkpoint": str(checkpoint_path),
            "predicted_class": checkpoint["class_names"][predicted_index],
            "probabilities": dict(zip(checkpoint["class_names"], map(float, probabilities))),
            "explanation_note": "Grad-CAM is qualitative and does not establish clinical validity.",
        }
        with (prediction_dir / "prediction.json").open("w", encoding="utf-8") as stream:
            json.dump(prediction_record, stream, indent=2)
        columns[1].image(str(prediction_dir / "gradcam_overlay.png"), caption="Grad-CAM overlay (qualitative)", use_container_width=True)
        st.subheader(f"Predicted class: {prediction_record['predicted_class']}")
        st.dataframe(pd.DataFrame({"Class": checkpoint["class_names"], "Probability": probabilities}).sort_values("Probability", ascending=False), hide_index=True, use_container_width=True)
        st.caption(f"Prediction artifacts saved to {prediction_dir}")

comparison_path = Path("outputs/comparisons/comparison.csv")
if comparison_path.is_file():
    st.subheader("Saved experiment comparison")
    comparisons = pd.read_csv(comparison_path)
    st.dataframe(comparisons, hide_index=True, use_container_width=True)
    plot_path = Path("outputs/comparisons/macro_f1_comparison.png")
    if plot_path.is_file():
        st.image(str(plot_path), use_container_width=True)