import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

def evaluate_fraud(y_true, y_pred, y_proba=None):
    """
    Fraud detection evaluation (binary classification, imbalanced).
    Assumes fraud label = 1.
    """

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    extras = {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]  # take fraud probability

        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))

    return metrics, extras
