# src/training/thresholds.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

def _ensure_1d(a):
    """Ensure a 1-D float numpy array."""
    return np.asarray(a).astype(float).ravel()

def _strategy_returns_long_flat(p: np.ndarray, threshold: float, returns_next: np.ndarray) -> np.ndarray:
    """
    Simple long/flat strategy:
    - Position = 1 if p >= threshold
    - Position = 0 otherwise
    PnL = position * next-period returns
    """
    pos = (p >= threshold).astype(float)
    return pos * returns_next

def _annualized_sharpe(r: np.ndarray, freq_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio from a return series.
    Assumes zero risk-free rate; for daily data use freq_per_year ~ 252.
    """
    r = _ensure_1d(r)
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(np.sqrt(freq_per_year) * mu / sigma)

@dataclass
class ThresholdSearchResult:
    best_threshold: float
    criterion_value: float
    table: pd.DataFrame

def grid_search_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    objective: str = "f1",
    returns_next: Optional[np.ndarray] = None,
    freq_per_year: int = 252,
) -> ThresholdSearchResult:
    """
    Search a threshold θ that maximizes:
      - "f1" (default)
      - "sharpe" (requires returns_next)
    Returns the per-threshold metrics table and the best θ & value.
    """
    y = _ensure_1d(y_true).astype(int)
    p = _ensure_1d(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)  # reasonably fine grid
    rows = []

    # Pre-compute threshold-free metrics for reference
    try:
        roc_auc = roc_auc_score(y, p)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y, p)
    except Exception:
        pr_auc = np.nan

    for th in thresholds:
        y_hat = (p >= th).astype(int)
        acc = accuracy_score(y, y_hat)
        f1 = f1_score(y, y_hat, pos_label=1, zero_division=0)
        prec = precision_score(y, y_hat, pos_label=1, zero_division=0)
        rec = recall_score(y, y_hat, pos_label=1, zero_division=0)

        sharp = np.nan
        if returns_next is not None:
            strat = _strategy_returns_long_flat(p, th, _ensure_1d(returns_next))
            sharp = _annualized_sharpe(strat, freq_per_year=freq_per_year)

        rows.append((th, acc, f1, prec, rec, roc_auc, pr_auc, sharp))

    df = pd.DataFrame(
        rows,
        columns=["threshold", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "sharpe"]
    )

    if objective == "f1":
        idx = int(df["f1"].values.argmax())
        best_val = float(df.loc[idx, "f1"])
    elif objective == "sharpe":
        assert returns_next is not None, "returns_next must be provided for objective='sharpe'."
        idx = int(df["sharpe"].values.argmax())
        best_val = float(df.loc[idx, "sharpe"])
    else:
        raise ValueError("objective must be one of {'f1', 'sharpe'}")

    best_th = float(df.loc[idx, "threshold"])
    return ThresholdSearchResult(best_threshold=best_th, criterion_value=best_val, table=df)

def plot_metric_vs_threshold(df: pd.DataFrame, metric: str = "f1", show: bool = True, savepath: Optional[str] = None) -> None:
    """
    Plot any metric present in the DataFrame against the threshold.
    Common choices: 'f1' or 'sharpe'.
    """
    assert metric in df.columns, f"{metric} not found in DataFrame columns."
    plt.figure(figsize=(6, 4))
    plt.plot(df["threshold"], df[metric], marker="o")
    plt.xlabel("Threshold θ")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs threshold")
    plt.grid(True, alpha=0.3)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
