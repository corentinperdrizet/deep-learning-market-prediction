# src/training/metrics.py
from __future__ import annotations
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def _safe_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return float("nan")


def evaluate_classifier_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Return a dict of standard classification metrics.

    y_proba: array (N,2) with columns [p0, p1]
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_proba)[:, 1]
    # Default threshold 0.5 for hard predictions
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(_safe_auc_roc(y_true, y_score)),
        "pr_auc": float(_safe_auc_pr(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }

    # Calibration summary via PR curve area shape (optional diagnostic)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # rough shape measure: mean precision
        metrics["pr_mean_precision"] = float(np.nanmean(precision))
    except Exception:
        metrics["pr_mean_precision"] = float("nan")

    return metrics


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    # R^2 can be negative; handle degenerate var
    var = float(np.var(y_true))
    r2 = float(1 - mse / var) if var > 1e-12 else float("nan")

    # Directional accuracy if we convert regression to sign
    sign_true = (y_true > 0).astype(int)
    sign_pred = (y_pred > 0).astype(int)
    directional_acc = float(np.mean(sign_true == sign_pred))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "directional_acc": directional_acc,
    }
