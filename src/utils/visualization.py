import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.config import LABEL_MAPPING, OUTPUT_DIR


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "MoveMint Transportation Mode Confusion Matrix"
):
    """Plots and saves normalized confusion matrix heatmap."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    labels = [LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Transit Mode", fontsize=11)
    ax.set_xlabel("Predicted Mode", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None
):
    """Plots and saves train/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    ax2.plot(history["train_acc"], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(history["val_acc"], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")
