import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
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

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Simple confusion matrix plot (no seaborn).
    cm must be a 2x2 array for binary fraud.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Not Fraud (0)", "Fraud (1)"])
    plt.yticks([0, 1], ["Not Fraud (0)", "Fraud (1)"])

    # Add counts
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()
    return fig


def plot_roc(y_true, y_proba, title="ROC Curve"):
    """
    ROC curve for fraud detection.
    y_proba can be (n,) or (n,2).
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.show()
    return fig


def plot_precision_recall(y_true, y_proba, title="Precision-Recall Curve"):
    """
    PR curve for fraud detection (more informative for imbalance).
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.show()
    return fig


def find_best_threshold_by_f1(y_true, y_proba):
    """
    Finds the threshold that maximizes F1.
    Returns: best_threshold, best_f1
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)


def save_eval_row_csv(model_name, metrics, out_path="results/evaluation_summary.csv"):
    """
    Appends a single evaluation row into results/evaluation_summary.csv.
    Creates folder if not exists.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    row = {"model": model_name, **metrics}

    import pandas as pd
    df_row = pd.DataFrame([row])

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(out_path, index=False)

def cross_validate_fraud(model, X, y, n_splits=5, random_state=42, threshold=None):
    """
    Stratified CV for binary fraud detection.
    Returns a dict with per-fold metrics + mean/std.

    threshold:
      - None: use model.predict()
      - float: use proba >= threshold as prediction
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        # Probabilities (preferred)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        # Prediction (either thresholded proba or standard predict)
        if (threshold is not None) and (y_proba is not None):
            y_proba_1 = np.asarray(y_proba)
            if y_proba_1.ndim == 2:
                y_proba_1 = y_proba_1[:, 1]
            y_pred = (y_proba_1 >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        metrics, _ = evaluate_fraud(y_test, y_pred, y_proba=y_proba)

        row = {"fold": fold, **metrics}
        fold_rows.append(row)

    import pandas as pd
    df = pd.DataFrame(fold_rows)

    mean_metrics = df.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    std_metrics = df.drop(columns=["fold"]).std(numeric_only=True).to_dict()

    return {
        "per_fold": df,
        "mean": {k: float(v) for k, v in mean_metrics.items()},
        "std": {k: float(v) for k, v in std_metrics.items()},
    }


def compare_models_cv(
    modelA, modelB, X, y,
    modelA_name="ModelA",
    modelB_name="ModelB",
    n_splits=5,
    random_state=42,
    out_path="results/week10_cv_comparison.csv",
    threshold=None
):

    resA = cross_validate_fraud(modelA, X, y, n_splits=n_splits, random_state=random_state, threshold=threshold)
    resB = cross_validate_fraud(modelB, X, y, n_splits=n_splits, random_state=random_state, threshold=threshold)

    rows = []

    for name, res in [(modelA_name, resA), (modelB_name, resB)]:
        row = {"model": name}
        for k, v in res["mean"].items():
            row[f"mean_{k}"] = v
        for k, v in res["std"].items():
            row[f"std_{k}"] = v
        rows.append(row)

    import pandas as pd
    df_out = pd.DataFrame(rows)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    return {
        "modelA": resA,
        "modelB": resB,
        "comparison_table": df_out,
        "saved_to": out_path
    }