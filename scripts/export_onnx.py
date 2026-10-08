import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from src.config import CHECKPOINT_DIR, OUTPUT_DIR, NUM_CHANNELS, WINDOW_LENGTH, NUM_CLASSES
from src.models.baseline_cnn import MoveMintEdgeCNN


def export_mobile_models(
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
    torchscript_path: Path = OUTPUT_DIR / "movemint_edge_model.ptl",
    onnx_path: Path = OUTPUT_DIR / "movemint_edge_model.onnx"
):
    print("Loading PyTorch model checkpoint...")
    model = MoveMintEdgeCNN(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, NUM_CHANNELS, WINDOW_LENGTH, requires_grad=False)

    # 1. Export to TorchScript (PyTorch Mobile optimized)
    print(f"Exporting to TorchScript for mobile deployment: {torchscript_path}...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(torchscript_path))
    ts_size_kb = torchscript_path.stat().st_size / 1024
    print(f"[SUCCESS] TorchScript export complete. File size: {ts_size_kb:.2f} KB")

    # 2. Export to ONNX if onnx is available
    try:
        import onnx
        print(f"Exporting to ONNX: {onnx_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["sensor_stream"],
            output_names=["mode_probabilities"],
            dynamic_axes={
                "sensor_stream": {0: "batch_size"},
                "mode_probabilities": {0: "batch_size"}
            }
        )
        onnx_size_kb = onnx_path.stat().st_size / 1024
        print(f"[SUCCESS] ONNX export complete. File size: {onnx_size_kb:.2f} KB")
    except Exception as e:
        print(f"[INFO] ONNX/onnxscript not installed ({e}). TorchScript (.ptl) is ready for mobile.")


if __name__ == "__main__":
    export_mobile_models()
