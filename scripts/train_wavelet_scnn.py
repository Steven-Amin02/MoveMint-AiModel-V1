import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.config import (
    DEVICE,
    BATCH_SIZE,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LABEL_MAPPING,
    NUM_CHANNELS,
    WINDOW_LENGTH,
    NUM_CLASSES
)
from src.data.dataset import (
    load_htc_data,
    get_stratified_splits,
    compute_class_weights,
    create_dataloaders
)
from src.data.augmentations import CompositeAugmentations
from src.models.wavelet_scnn import MoveMintWaveletSCNN
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_confusion_matrix, plot_training_history


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_b).sum().item()
        total += len(y_b)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        logits = model(x_b)
        loss = criterion(logits, y_b)
        total_loss += loss.item() * len(y_b)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_b.cpu().numpy())
        total += len(y_b)
    return total_loss / total, np.array(all_targets), np.array(all_preds)


def run_wavelet_scnn_pipeline(epochs: int = 30, lr: float = 1e-3):
    print("=" * 70)
    print(" TRAINING WAVELET MULTI-SCALE S-CNN (W-SCNN) - MOVEMINT")
    print("=" * 70)
    print(f"Target Device: {DEVICE}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load Data
    x, y = load_htc_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_stratified_splits(x, y)
    print(f"Data Split: Train={len(y_train)} | Val={len(y_val)} | Test={len(y_test)}")

    # 2. DataLoaders with Augmentations
    train_transform = CompositeAugmentations()
    train_loader, val_loader, test_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=BATCH_SIZE,
        train_transform=train_transform
    )

    # 3. Model Setup
    model = MoveMintWaveletSCNN(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES, dropout_rate=0.3).to(DEVICE)
    summary = model.get_model_summary()
    print(f"Trainable Parameters: {summary['total_trainable_parameters']:,} ({summary['estimated_fp32_size_kb']} KB)")

    class_weights = compute_class_weights(y_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    checkpoint_path = CHECKPOINT_DIR / "best_wavelet_scnn.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining Wavelet S-CNN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_targets, val_preds = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        val_metrics = evaluate_predictions(val_targets, val_preds)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"Car FPR: {val_metrics['anti_cheat_car_to_transit_fpr']*100:.2f}%"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
                "anti_cheat_car_to_transit_fpr": val_metrics["anti_cheat_car_to_transit_fpr"]
            }, checkpoint_path)

    print("-" * 70)
    print(f"Best Validation Macro-F1: {best_val_f1*100:.2f}%. Saved to {checkpoint_path}")

    # 4. Final Evaluation on Held-Out Test Set
    print("\nEvaluating Best Model on Held-Out Test Set (1,500 unseen windows)...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_targets, test_preds = evaluate(model, test_loader, criterion, DEVICE)
    test_metrics = evaluate_predictions(test_targets, test_preds)

    print("=" * 70)
    print(" WAVELET MULTI-SCALE S-CNN (W-SCNN) TEST RESULTS")
    print("=" * 70)
    print(f"Overall Test Accuracy:                  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:                         {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:                      {test_metrics['weighted_f1']*100:.2f}%")
    print(f"Anti-Cheat Car-to-Transit FPR:          {test_metrics['anti_cheat_car_to_transit_fpr']*100:.2f}% (Target: <= 5%)")
    print(f"Public Transit Verification Recall:     {test_metrics['anti_cheat_transit_recall']*100:.2f}% (Target: >= 90%)")
    print(f"Micro-Mobility Recall:                  {test_metrics['anti_cheat_micro_mobility_recall']*100:.2f}%")
    print("=" * 70)

    # Per-Class Summary
    report = test_metrics["classification_report"]
    print("\nPER-CLASS REPORT:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 52)
    for c_id in range(len(LABEL_MAPPING)):
        name = LABEL_MAPPING[c_id]
        c_stats = report[name]
        print(f"{name:<12} {c_stats['precision']*100:>8.2f}% {c_stats['recall']*100:>8.2f}% {c_stats['f1-score']*100:>8.2f}% {int(c_stats['support']):>9d}")

    # Visualizations
    plot_training_history(history, save_path=OUTPUT_DIR / "wavelet_scnn_curves.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        save_path=OUTPUT_DIR / "wavelet_scnn_confusion_matrix.png",
        title="Wavelet Multi-Scale S-CNN Confusion Matrix"
    )

    # 5. Export Mobile Models (TorchScript & ONNX)
    model.eval().to("cpu")
    dummy = torch.randn(1, NUM_CHANNELS, WINDOW_LENGTH, requires_grad=False)
    
    # TorchScript
    ts_path = OUTPUT_DIR / "movemint_wavelet_scnn.ptl"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))
    print(f"\n[EXPORT] TorchScript saved: {ts_path} ({ts_path.stat().st_size / 1024:.2f} KB)")

    # ONNX
    onnx_path = OUTPUT_DIR / "movemint_wavelet_scnn.onnx"
    try:
        import onnx
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["sensor_stream"],
            output_names=["mode_probabilities"],
            dynamic_axes={"sensor_stream": {0: "batch_size"}, "mode_probabilities": {0: "batch_size"}}
        )
        print(f"[EXPORT] ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)")
        edge_onnx_path = OUTPUT_DIR / "movemint_edge_model.onnx"
        shutil.copyfile(onnx_path, edge_onnx_path)
        print(f"[EXPORT] Synchronized: {edge_onnx_path}")
    except Exception as e:
        print(f"[INFO] ONNX export note: {e}")

    return test_metrics


if __name__ == "__main__":
    run_wavelet_scnn_pipeline(epochs=30)
