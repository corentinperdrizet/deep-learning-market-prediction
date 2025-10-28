# src/training/evaluate.py
from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_classifier_metrics
from ..models.baselines import (
    BuyAndHoldClassifier,
    LogisticRegressionTabular,
    XGBTabular,
    SMACrossoverClassifier,
    sequences_to_tabular,
)


def evaluate_classifier(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Thin wrapper for external use."""
    return evaluate_classifier_metrics(y_true, y_proba)


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from .metrics import evaluate_regression_metrics

    return evaluate_regression_metrics(y_true, y_pred)


def run_baselines(
    dataset: Dict[str, Any],
    prices: Optional[pd.Series] = None,
    use_xgb: bool = False,
    pooling_lr: str = "last",
) -> pd.DataFrame:
    """
    Train/evaluate Buy&Hold, LR (and optional XGB) on val/test.
    Optionally include SMA(50/200) if prices provided (aligned to dataset index).

    Returns a tidy DataFrame with rows=method/split and columns=metrics.
    """
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"].astype(int)
    y_val = dataset["y_val"].astype(int)
    y_test = dataset["y_test"].astype(int)

    # Build an index for alignment (assumes dataset['idx'] provides slices or indices)
    # Fallback: create a simple RangeIndex if not available
    idx = dataset.get("idx", {})
    index_val = idx.get("val")  # may be slice or pandas index
    index_test = idx.get("test")

    # If the pipeline doesn't provide actual pandas indices, we create dummy ones
    if isinstance(index_val, (np.ndarray, list)):
        index_val = pd.Index(index_val)
    elif index_val is None:
        index_val = pd.RangeIndex(start=0, stop=len(y_val))

    if isinstance(index_test, (np.ndarray, list)):
        index_test = pd.Index(index_test)
    elif index_test is None:
        index_test = pd.RangeIndex(start=0, stop=len(y_test))

    rows = []

    # 1) Buy & Hold probabilistic baseline (prevalence)
    bh = BuyAndHoldClassifier().fit(y_train)
    for split_name, y_split in [("val", y_val), ("test", y_test)]:
        proba = bh.predict_proba(len(y_split))
        m = evaluate_classifier(y_split, proba)
        rows.append({"model": "buy_hold", "split": split_name, **m})

    # 2) Logistic Regression on tabularized features
    lr = LogisticRegressionTabular(pooling=pooling_lr).fit(X_train, y_train)
    for split_name, X_split, y_split in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = lr.predict_proba(X_split)
        m = evaluate_classifier(y_split, proba)
        rows.append({"model": f"logreg[{pooling_lr}]", "split": split_name, **m})

    # 3) Optional XGB
    if use_xgb:
        xgb = XGBTabular(pooling="flatten_last_k", k=5).fit(X_train, y_train)
        for split_name, X_split, y_split in [("val", X_val, y_val), ("test", X_test, y_test)]:
            proba = xgb.predict_proba(X_split)
            m = evaluate_classifier(y_split, proba)
            rows.append({"model": "xgb[flat_k=5]", "split": split_name, **m})

    # 4) SMA crossover (if prices provided)
    if prices is not None:
        sma = SMACrossoverClassifier(50, 200).fit(prices)
        for split_name, y_split, idx_split in [
            ("val", y_val, index_val),
            ("test", y_test, index_test),
        ]:
            proba = sma.predict_proba(idx_split)
            # Ensure lengths match (may need alignment/trim)
            n = min(len(proba), len(y_split))
            m = evaluate_classifier(y_split[:n], proba[:n])
            rows.append({"model": "sma_50_200", "split": split_name, **m})

    df = pd.DataFrame(rows)
    # Optional ordering of columns
    preferred = [
        "model",
        "split",
        "accuracy",
        "f1_pos",
        "roc_auc",
        "pr_auc",
        "brier",
        "pr_mean_precision",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]
