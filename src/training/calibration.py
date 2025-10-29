# src/training/calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from joblib import dump, load
import matplotlib.pyplot as plt

EPS = 1e-12

def _clip_probs(p: np.ndarray) -> np.ndarray:
    """Numerically safe clipping for probabilities."""
    return np.clip(p, EPS, 1.0 - EPS)

def _logit(p: np.ndarray) -> np.ndarray:
    """Logit transform with clipping."""
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))

@dataclass
class PlattCalibrator:
    """
    Lightweight Platt scaling:
    learns a sigmoid transformation on validation probabilities.
    Form: p' = sigmoid(a * logit(p) + b)
    """
    a: Optional[float] = None
    b: Optional[float] = None

    def fit(self, y_true: np.ndarray, p_val: np.ndarray) -> "PlattCalibrator":
        """
        Fit 'a' and 'b' by maximizing Bernoulli log-likelihood via Newton-Raphson.
        This avoids refitting the base model; we calibrate its outputs instead.
        """
        y = y_true.astype(np.float64).ravel()
        z = _logit(p_val.ravel())

        # Initialization: start from prior log-odds for 'b', zero slope for 'a'
        a, b = 0.0, np.log((y.mean() + EPS) / (1 - y.mean() + EPS))
        for _ in range(50):
            t = a * z + b
            s = 1.0 / (1.0 + np.exp(-t))  # sigmoid
            # Gradient
            g_a = np.sum((y - s) * z)
            g_b = np.sum(y - s)
            # Hessian
            w = s * (1 - s)
            h_aa = np.sum(w * z * z)
            h_ab = np.sum(w * z)
            h_bb = np.sum(w)
            # Solve the 2x2 Newton system with a tiny jitter
            H = np.array([[h_aa, h_ab],
                          [h_ab, h_bb]]) + np.eye(2) * 1e-8
            g = np.array([g_a, g_b])
            step = np.linalg.solve(H, g)
            a += step[0]
            b += step[1]
            if np.linalg.norm(step) < 1e-8:
                break

        self.a, self.b = float(a), float(b)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Apply the learned sigmoid mapping to raw probabilities."""
        assert self.a is not None and self.b is not None, "Call fit(...) before transform(...)."
        z = _logit(p)
        t = self.a * z + self.b
        return 1.0 / (1.0 + np.exp(-t))

    def save(self, path: str) -> None:
        """Persist the calibrator parameters."""
        dump({"a": self.a, "b": self.b}, path)

    @staticmethod
    def load(path: str) -> "PlattCalibrator":
        """Load a saved PlattCalibrator."""
        obj = load(path)
        return PlattCalibrator(a=obj["a"], b=obj["b"])


@dataclass
class IsotonicCalibrator:
    """
    Non-parametric monotonic calibration using isotonic regression.
    """
    iso: Optional[IsotonicRegression] = None

    def fit(self, y_true: np.ndarray, p_val: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on validation probabilities."""
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(p_val.ravel(), y_true.astype(np.float64).ravel())
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the learned isotonic function."""
        assert self.iso is not None, "Call fit(...) before transform(...)."
        return np.asarray(self.iso.transform(p.ravel())).reshape(p.shape)

    def save(self, path: str) -> None:
        """Persist the isotonic model."""
        dump(self.iso, path)

    @staticmethod
    def load(path: str) -> "IsotonicCalibrator":
        """Load a saved IsotonicCalibrator."""
        iso = load(path)
        return IsotonicCalibrator(iso=iso)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[float, float, pd.DataFrame]:
    """
    Compute ECE (weighted mean |accuracy - confidence|) and MCE (max gap).
    Returns the per-bin calibration table as well.
    """
    y = y_true.astype(np.float64).ravel()
    p = p.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append((b, 0, np.nan, np.nan))
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        rows.append((b, m.sum(), acc, conf))

    df = pd.DataFrame(rows, columns=["bin", "count", "accuracy", "confidence"])
    df["gap"] = np.abs(df["accuracy"] - df["confidence"])
    weights = df["count"] / max(1, df["count"].sum())
    ece = np.nansum(weights * df["gap"])
    mce = np.nanmax(df["gap"])
    return float(ece), float(mce), df


def calibration_report(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Convenience wrapper to compute ECE/MCE and Brier score on probabilities.
    """
    p = _clip_probs(p)
    ece, mce, _ = expected_calibration_error(y_true, p, n_bins)
    brier = brier_score_loss(y_true, p)
    return {"ECE": ece, "MCE": mce, "Brier": float(brier)}


def plot_reliability_diagram(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration curve",
    show: bool = True,
    savepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot reliability diagram: empirical accuracy vs mean predicted probability per bin.
    Returns the underlying calibration table (non-empty bins).
    """
    _, _, table = expected_calibration_error(y_true, p, n_bins)
    tbl = table.dropna(subset=["accuracy", "confidence"])

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(tbl["confidence"], tbl["accuracy"], marker="o")
    plt.xlabel("Confidence (mean predicted probability)")
    plt.ylabel("Empirical accuracy (positives rate)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return tbl
