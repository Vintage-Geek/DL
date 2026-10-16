"""ResNet-50 factory and checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    if num_classes < 2:
        raise ValueError("ResNet-50 classification requires at least two classes.")
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint has no class_names metadata: {checkpoint_path}")
    model = build_resnet50(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint