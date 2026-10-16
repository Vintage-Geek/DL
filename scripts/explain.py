"""Generate a Grad-CAM overlay and prediction record for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.common import load_config
from src.explainability import GradCAM, save_gradcam_overlay
from src.models import load_checkpoint
from src.preprocessing import PreprocessingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/explanations")
    arguments = parser.parse_args()
    run_dir = Path(arguments.run_dir)
    image_path = Path(arguments.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    config = load_config(run_dir / "config.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    method = config["experiment"]["preprocessing"]
    dataset_config = config["dataset"]
    image_size = int(config["data"].get("image_size", 224))
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    processed = PreprocessingPipeline(method, bool(dataset_config.get("grayscale_compatible", False)))(resized)
    tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
    if config["data"].get("imagenet_normalization", True):
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        standard_deviation = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = (tensor - mean) / standard_deviation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device=device)
    explainer = GradCAM(model)
    heatmap, predicted_index = explainer(tensor.unsqueeze(0).to(device))
    explainer.close()
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().tolist()
    output = Path(arguments.output_dir) / f"{run_dir.parent.parent.name}_{method}_{image_path.stem}"
    output.mkdir(parents=True, exist_ok=True)
    original_output = output / "original.png"
    Image.fromarray(image_rgb).save(original_output)
    overlay_output = output / "gradcam_overlay.png"
    save_gradcam_overlay(image_rgb, heatmap, overlay_output)
    prediction = {
        "source_image": str(image_path),
        "preprocessing": method,
        "predicted_class": checkpoint["class_names"][predicted_index],
        "class_probabilities": dict(zip(checkpoint["class_names"], probabilities)),
        "explanation_note": "Grad-CAM is qualitative and does not establish clinical validity.",
    }
    with (output / "prediction.json").open("w", encoding="utf-8") as stream:
        json.dump(prediction, stream, indent=2)
    print(json.dumps({"output_dir": str(output), **prediction}, indent=2))


if __name__ == "__main__":
    main()