import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LABEL_MAPPING
)
from src.data.dataset import (
    load_htc_data,
    get_stratified_splits,
    compute_class_weights,
    create_dataloaders
)
from src.data.augmentations import CompositeAugmentations
from src.models.baseline_cnn import MoveMintEdgeCNN
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_confusion_matrix, plot_training_history


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = y_batch.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)
        total += len(y_batch)

    avg_loss = total_loss / total
    return avg_loss, np.array(all_targets), np.array(all_preds)


def run_training_pipeline(epochs: int = EPOCHS) -> Dict[str, Any]:
    print("=" * 70)
    print(" MOVEMINT EDGE-AI PIPELINE: TRAINING & BENCHMARKING (PHASE 1)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load Data
    print("Loading HTC sensor dataset...")
    x, y = load_htc_data()
    print(f"Loaded X shape: {x.shape}, Y shape: {y.shape}")

    # 2. Stratified Splits
    print("Partitioning data into Stratified Train (70%), Val (15%), Test (15%)...")
    x_train, y_train, x_val, y_val, x_test, y_test = get_stratified_splits(x, y)
    print(f"Train samples: {len(y_train)} | Val samples: {len(y_val)} | Test samples: {len(y_test)}")

    # 3. Class Weights for Loss
    class_weights = compute_class_weights(y_train).to(DEVICE)
    print(f"Computed Class Weights: {np.round(class_weights.cpu().numpy(), 2)}")

    # 4. DataLoaders with Augmentations
    train_transform = CompositeAugmentations()
    train_loader, val_loader, test_loader = create_dataloaders(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        batch_size=BATCH_SIZE,
        train_transform=train_transform
    )

    # 5. Model Initialization
    model = MoveMintEdgeCNN(in_channels=12, num_classes=8).to(DEVICE)
    model_summary = model.get_model_summary()
    print("-" * 70)
    print(f"Model Summary: {model_summary['total_trainable_parameters']:,} parameters")
    print(f"Estimated Model Size: {model_summary['estimated_fp32_size_kb']} KB (FP32) | {model_summary['estimated_int8_size_kb']} KB (INT8)")
    print("-" * 70)

    # 6. Optimizer, Scheduler, Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_f1 = 0.0
    best_checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    # 7. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_targets, val_preds = evaluate_epoch(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step()

        val_metrics = evaluate_predictions(val_targets, val_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"Car FPR: {val_metrics['anti_cheat_car_to_transit_fpr']*100:.2f}%"
        )

        # Checkpointing based on Validation Macro-F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
                "anti_cheat_car_to_transit_fpr": val_metrics["anti_cheat_car_to_transit_fpr"]
            }, best_checkpoint_path)

    print("-" * 70)
    print(f"Best Validation Macro-F1: {best_val_f1*100:.2f}%. Checkpoint saved to {best_checkpoint_path}")

    # 8. Final Evaluation on Held-Out Test Set
    print("\nEvaluating on Held-Out Test Set (1,500 unseen windows)...")
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_targets, test_preds = evaluate_epoch(
        model, test_loader, criterion, DEVICE
    )
    test_metrics = evaluate_predictions(test_targets, test_preds)

    print("=" * 70)
    print(" HELD-OUT TEST RESULTS & MOVEMINT SECURITY METRICS")
    print("=" * 70)
    print(f"Overall Test Accuracy:                  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:                         {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:                      {test_metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Precision:                        {test_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:                           {test_metrics['macro_recall']*100:.2f}%")
    print("-" * 70)
    print("MOVEMINT OPERATIONAL METRICS:")
    print(f">> Anti-Cheat Car-to-Transit FPR:      {test_metrics['anti_cheat_car_to_transit_fpr']*100:.2f}% (Target: <= 5%)")
    print(f">> Public Transit Verification Recall: {test_metrics['anti_cheat_transit_recall']*100:.2f}% (Target: >= 90%)")
    print(f">> Micro-Mobility Recall:              {test_metrics['anti_cheat_micro_mobility_recall']*100:.2f}% (Walk/Run/Bike)")
    print("=" * 70)

    # Print Per-Class Performance
    print("\nPER-CLASS CLASSIFICATION REPORT:")
    report = test_metrics["classification_report"]
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 52)
    for c_id in range(len(LABEL_MAPPING)):
        name = LABEL_MAPPING[c_id]
        c_stats = report[name]
        print(
            f"{name:<12} "
            f"{c_stats['precision']*100:>8.2f}% "
            f"{c_stats['recall']*100:>8.2f}% "
            f"{c_stats['f1-score']*100:>8.2f}% "
            f"{int(c_stats['support']):>9d}"
        )

    # 9. Visualizations
    plot_training_history(history, save_path=OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], save_path=OUTPUT_DIR / "test_confusion_matrix.png")

    return {
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(best_checkpoint_path)
    }


if __name__ == "__main__":
    from typing import Tuple
    run_training_pipeline()
