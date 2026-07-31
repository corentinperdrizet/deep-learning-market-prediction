# src/app/utils.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_json_if_exists(path: Path | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def find_figure_if_exists(figures_dir: Path, name: str | None) -> Path | None:
    if not name:
        return None
    p = figures_dir / name
    return p if p.exists() else None


def find_processed_parquet(processed_dir: Path) -> Path | None:
    if not processed_dir.exists():
        return None
    # Heuristic: take the first *_dataset.parquet if available
    candidates = sorted(processed_dir.glob("*_dataset.parquet"))
    if candidates:
        return candidates[0]
    # Otherwise, any parquet
    any_pq = sorted(processed_dir.glob("*.parquet"))
    return any_pq[0] if any_pq else None


@st.cache_data(show_spinner=False)
def load_parquet_head_if_exists(path: Path | None, n: int = 500) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) > n:
            return df.head(n)
        return df
    except Exception:
        return None


def _exists(p: Path | None) -> bool:
    return bool(p) and Path(p).exists()


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None


def discover_available_runs(artifacts_dir: Path, figures_dir: Path) -> dict[str, dict]:
    """
    Heuristic: if LSTM artifacts are found → exposes an “LSTM run”.
                 if Transformer artifacts are found → exposes a “run Transformer”.
    We then map to known files (reports, logs, figures, kpis, signals).
    """
    runs: dict[str, dict] = {}

    # ---------- LSTM ----------
    lstm_test_report = artifacts_dir / "lstm_test_report.json"
    lstm_logs = artifacts_dir / "lstm_logs.csv"
    lstm_kpis = artifacts_dir / "lstm_backtest_kpis.csv"
    lstm_signals = artifacts_dir / "lstm_signals.csv"  # optional
    lstm_importance_csv = artifacts_dir / "lstm_feature_importance.csv"
    lstm_loss_fig = "lstm_loss.png"
    lstm_metrics_fig = "lstm_metrics.png"
    lstm_lr_fig = "lstm_lr.png"
    lstm_equity_fig = "lstm_equity.png"
    lstm_dd_fig = "lstm_drawdown.png"
    lstm_importance_fig = "lstm_feature_importance.png"

    has_any_lstm = any(
        p.exists() for p in [lstm_test_report, lstm_logs, artifacts_dir / "lstm_classifier.pt"]
    ) or any(
        (figures_dir / n).exists()
        for n in [lstm_loss_fig, lstm_metrics_fig, lstm_lr_fig, lstm_equity_fig, lstm_dd_fig]
    )

    if has_any_lstm:
        runs["LSTM"] = {
            "test_report": lstm_test_report if lstm_test_report.exists() else None,
            "logs_csv": lstm_logs if lstm_logs.exists() else None,
            "kpis_csv": lstm_kpis if lstm_kpis.exists() else None,
            "signals_csv": lstm_signals if lstm_signals.exists() else None,
            "loss_fig": lstm_loss_fig,
            "metrics_fig": lstm_metrics_fig,
            "lr_fig": lstm_lr_fig,
            "equity_fig": lstm_equity_fig,
            "dd_fig": lstm_dd_fig,
            "importance_csv": lstm_importance_csv if lstm_importance_csv.exists() else None,
            "importance_fig": lstm_importance_fig,
            "attention_fig": None,
        }

    # ---------- Transformer ----------
    tr_test_report = artifacts_dir / "transformer_test_report.json"
    tr_logs = artifacts_dir / "transformer_logs.csv"
    tr_kpis = artifacts_dir / "transformer_backtest_kpis.csv"
    tr_signals = artifacts_dir / "transformer_signals.csv"  # optional
    tr_importance_csv = artifacts_dir / "transformer_feature_importance.csv"
    tr_loss_fig = "transformer_loss.png"
    tr_metrics_fig = "transformer_metrics.png"
    tr_lr_fig = "transformer_lr.png"
    tr_equity_fig = "transformer_equity.png"
    tr_dd_fig = "transformer_drawdown.png"
    tr_importance_fig = "transformer_feature_importance.png"
    tr_attention_fig = "transformer_attention.png"

    has_any_tr = any(
        p.exists() for p in [tr_test_report, tr_logs, artifacts_dir / "transformer_classifier.pt"]
    ) or any(
        (figures_dir / n).exists() for n in [tr_loss_fig, tr_metrics_fig, tr_lr_fig, tr_equity_fig, tr_dd_fig]
    )

    if has_any_tr:
        runs["Transformer"] = {
            "test_report": tr_test_report if tr_test_report.exists() else None,
            "logs_csv": tr_logs if tr_logs.exists() else None,
            "kpis_csv": tr_kpis if tr_kpis.exists() else None,
            "signals_csv": tr_signals if tr_signals.exists() else None,
            "loss_fig": tr_loss_fig,
            "metrics_fig": tr_metrics_fig,
            "lr_fig": tr_lr_fig,
            "equity_fig": tr_equity_fig,
            "dd_fig": tr_dd_fig,
            "importance_csv": tr_importance_csv if tr_importance_csv.exists() else None,
            "importance_fig": tr_importance_fig,
            "attention_fig": tr_attention_fig,
        }

    return runs


def compute_confusion_matrix(y_true: np.ndarray, y_hat: np.ndarray):
    """
    Returns a small 2x2 DataFrame + a textual summary.
    """
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    tp = int(((y_true == 1) & (y_hat == 1)).sum())

    df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"],
    )
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    summary = f"Accuracy={acc:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}"
    return df, summary
