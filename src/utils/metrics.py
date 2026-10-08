import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any
from src.config import (
    LABEL_MAPPING,
    PRIVATE_VEHICLE_CLASSES,
    PUBLIC_TRANSIT_CLASSES,
    MICRO_MOBILITY_CLASSES
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class metrics along with specialized
    MoveMint anti-cheat operational metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_MAPPING))))

    # MoveMint Anti-Cheat Security Metric: Car falsely claimed as Public Transit
    car_mask = np.isin(y_true, PRIVATE_VEHICLE_CLASSES)  # True Cars
    total_cars = np.sum(car_mask)
    if total_cars > 0:
        car_pred_as_transit = np.sum(np.isin(y_pred[car_mask], PUBLIC_TRANSIT_CLASSES))
        car_to_transit_fpr = float(car_pred_as_transit / total_cars)
    else:
        car_to_transit_fpr = 0.0

    # Public Transit Verification Recall (Bus, Train, Subway)
    transit_mask = np.isin(y_true, PUBLIC_TRANSIT_CLASSES)
    total_transit = np.sum(transit_mask)
    if total_transit > 0:
        transit_verified = np.sum(np.isin(y_pred[transit_mask], PUBLIC_TRANSIT_CLASSES))
        transit_recall = float(transit_verified / total_transit)
    else:
        transit_recall = 0.0

    # Micro-Mobility Recall (Walk, Run, Bike)
    micro_mask = np.isin(y_true, MICRO_MOBILITY_CLASSES)
    total_micro = np.sum(micro_mask)
    if total_micro > 0:
        micro_verified = np.sum(np.isin(y_pred[micro_mask], MICRO_MOBILITY_CLASSES))
        micro_mobility_recall = float(micro_verified / total_micro)
    else:
        micro_mobility_recall = 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "anti_cheat_car_to_transit_fpr": car_to_transit_fpr,
        "anti_cheat_transit_recall": transit_recall,
        "anti_cheat_micro_mobility_recall": micro_mobility_recall,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABEL_MAPPING))),
            target_names=[LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))],
            output_dict=True,
            zero_division=0
        )
    }
