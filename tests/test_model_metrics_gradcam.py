import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_metrics
from src.explainability.gradcam import GradCAM
from src.models.resnet import build_resnet50


def test_metrics_include_macro_and_per_class_values():
    report = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert report["per_class"]["a"]["support"] == 2
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]


def test_resnet50_output_shape():
    model = build_resnet50(3).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 3)


def test_gradcam_outputs_finite_heatmap():
    model = build_resnet50(2).eval()
    explainer = GradCAM(model)
    heatmap, predicted = explainer(torch.randn(1, 3, 64, 64))
    explainer.close()
    assert heatmap.ndim == 2
    assert np.isfinite(heatmap).all()
    assert 0 <= predicted < 2