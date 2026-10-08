import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.config import WINDOW_LENGTH, NUM_CHANNELS, NUM_CLASSES
from src.data.augmentations import CompositeAugmentations, Jitter, Scaling, TimeMask, Random3DRotation
from src.models.baseline_cnn import MoveMintEdgeCNN
from src.utils.metrics import evaluate_predictions


def test_augmentations():
    x = np.random.randn(WINDOW_LENGTH, NUM_CHANNELS).astype(np.float32)
    aug = CompositeAugmentations()
    x_aug = aug(x)

    assert x_aug.shape == x.shape, f"Augmentation changed shape: {x_aug.shape} vs {x.shape}"
    assert not np.isnan(x_aug).any(), "Augmentation introduced NaNs"
    assert not np.isinf(x_aug).any(), "Augmentation introduced Infs"
    print("[PASS] Augmentations test passed.")


def test_model_forward():
    model = MoveMintEdgeCNN(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()

    dummy_input = torch.randn(4, NUM_CHANNELS, WINDOW_LENGTH)  # (B=4, C=12, T=450)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
    print("[PASS] Model forward pass test passed.")


def test_model_parameter_count():
    model = MoveMintEdgeCNN(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    params = model.count_parameters()
    # Ensure parameter count stays under 100K parameters for mobile edge deployment
    assert params < 100000, f"Model has too many parameters: {params}"
    print(f"[PASS] Model parameter check passed ({params:,} parameters).")


def test_metrics():
    y_true = np.array([4, 4, 4, 5, 5, 1, 0])
    y_pred = np.array([4, 5, 4, 5, 5, 1, 0])  # One Car predicted as Bus (5)

    metrics = evaluate_predictions(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert abs(metrics["anti_cheat_car_to_transit_fpr"] - (1 / 3)) < 1e-4
    assert abs(metrics["anti_cheat_transit_recall"] - 1.0) < 1e-4
    print("[PASS] Metrics evaluation test passed.")


if __name__ == "__main__":
    test_augmentations()
    test_model_forward()
    test_model_parameter_count()
    test_metrics()
    print("\nALL PIPELINE TESTS PASSED SUCCESSFULLY!")
