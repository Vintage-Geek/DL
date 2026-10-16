"""Grad-CAM for supported ResNet-50 convolutional layers."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or model.layer4[-1]
        self.activations = None
        self.gradients = None
        self._activation_hook = self.target_layer.register_forward_hook(self._capture_activation)
        self._gradient_hook = self.target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, module, inputs, output):
        self.activations = output

    def _capture_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        predicted_index = int(logits.argmax(dim=1).item())
        target_index = predicted_index if class_index is None else class_index
        if target_index < 0 or target_index >= logits.shape[1]:
            raise ValueError(f"class_index must be between 0 and {logits.shape[1] - 1}.")
        logits[:, target_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).relu()[0]
        heatmap = heatmap.detach().cpu().numpy()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        return heatmap.astype(np.float32), predicted_index

    def close(self) -> None:
        self._activation_hook.remove()
        self._gradient_hook.remove()


def save_gradcam_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.4) -> None:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with three channels.")
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored, alpha, 0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, overlay)