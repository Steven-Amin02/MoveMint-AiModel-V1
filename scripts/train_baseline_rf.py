import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.config import HTC_X_PATH, HTC_Y_PATH, RANDOM_SEED, OUTPUT_DIR
from src.data.dataset import load_htc_data, get_stratified_splits
from src.models.baseline_features import extract_features_dataset
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_confusion_matrix


def run_random_forest_baseline():
    print("=" * 70)
    print(" MOVEMINT CLASSICAL ML BASELINE: RANDOM FOREST (14 FEATURES)")
    print("=" * 70)

    # 1. Load Data
    x, y = load_htc_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_stratified_splits(x, y)

    # 2. Extract 14 Features
    print("Extracting 14 physical/frequency features for Train set...")
    x_train_feats = extract_features_dataset(x_train)
    print("Extracting 14 features for Test set...")
    x_test_feats = extract_features_dataset(x_test)

    # 3. Fit Random Forest
    print("Fitting RandomForestClassifier (n_estimators=100)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(x_train_feats, y_train)

    # 4. Evaluate on Test set
    y_test_pred = rf.predict(x_test_feats)
    metrics = evaluate_predictions(y_test, y_test_pred)

    print("-" * 70)
    print(f"Random Forest Test Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"Random Forest Macro F1:                 {metrics['macro_f1']*100:.2f}%")
    print(f"Anti-Cheat Car-to-Transit FPR:          {metrics['anti_cheat_car_to_transit_fpr']*100:.2f}%")
    print(f"Transit Verification Recall:            {metrics['anti_cheat_transit_recall']*100:.2f}%")
    print("=" * 70)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=OUTPUT_DIR / "rf_baseline_confusion_matrix.png",
        title="Random Forest (14 Features) Confusion Matrix"
    )
    return metrics


if __name__ == "__main__":
    run_random_forest_baseline()
