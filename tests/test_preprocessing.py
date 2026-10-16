import numpy as np
import pytest

from src.preprocessing import METHODS, preprocess_image


@pytest.mark.parametrize("method", METHODS)
def test_methods_return_valid_rgb_array(method):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[4:20, 6:26] = (180, 80, 30)
    result = preprocess_image(image, method, grayscale_compatible=(method == "P3"))
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_grayscale_method_requires_explicit_compatibility():
    with pytest.raises(ValueError, match="opt-in"):
        preprocess_image(np.zeros((8, 8, 3), dtype=np.uint8), "P3")


def test_z_score_is_per_channel_standardized():
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    result = preprocess_image(image, "P9")
    assert np.allclose(result.mean(axis=(0, 1)), 0, atol=1e-5)
    assert np.allclose(result.std(axis=(0, 1)), 1, atol=1e-5)


def test_unknown_method_fails_helpfully():
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        preprocess_image(np.zeros((4, 4), dtype=np.uint8), "PX")