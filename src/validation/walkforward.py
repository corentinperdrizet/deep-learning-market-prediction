# -----------------------------
# File: src/validation/walkforward.py
# -----------------------------
"""Walk-forward (expanding-window, purged) cross-validation.

The main pipeline (src/data/dataset.py::prepare_dataset) evaluates a model on
a *single* chronological train/val/test split. That answers "does this model
beat chance on this one test window?" but says nothing about how sensitive
the result is to which window was picked -- a single split can look good (or
bad) by luck. Walk-forward CV re-answers the question across several
consecutive out-of-sample windows, each preceded by an *embargo* gap so that
a label computed from a horizon-h-ahead return can never leak across the
train/test boundary of a fold (the single-split pipeline doesn't need this
because it only has one boundary and horizon is small relative to the val
gap; walk-forward has many boundaries, so the embargo has to be explicit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..backtest.engine import backtest
from ..backtest.metrics import summary_kpis
from ..backtest.rules import signal_from_proba
from ..data.config import DataConfig
from ..data.dataset import build_feature_frame
from ..data.preprocessing import FEATURE_COLUMNS_DEFAULT
from ..data.scaling import fit_scaler, transform_with_scaler
from ..data.sequences import build_sequences
from ..models.baselines import LogisticRegressionTabular
from ..models.lstm import LSTMClassifier
from ..training.dataloaders import make_loaders
from ..training.trainer import TrainConfig, fit
from ..training.utils import ensure_dir, get_device, set_global_seed


@dataclass
class Fold:
    index: int
    train_idx: pd.Index
    test_idx: pd.Index


@dataclass
class WalkForwardSplitter:
    """Expanding-window splitter with a purge/embargo gap between train and test.

    Fold 0 is the earliest (smallest train set), fold n_splits-1 the latest
    (largest train set, most recent test window) -- standard walk-forward
    ordering, train sets only ever grow.
    """

    n_splits: int = 5
    min_train_size: int = 500
    test_size: int = 90
    embargo: int = 1

    def split(self, data: pd.DataFrame) -> list[Fold]:
        n = len(data)
        raw_folds = []
        for k in range(self.n_splits):
            test_end = n - k * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.embargo
            if test_start < 0 or train_end < self.min_train_size:
                break
            raw_folds.append((train_end, test_start, test_end))
        raw_folds.reverse()  # chronological: oldest fold first

        folds = []
        for i, (train_end, test_start, test_end) in enumerate(raw_folds):
            folds.append(
                Fold(
                    index=i,
                    train_idx=data.index[0:train_end],
                    test_idx=data.index[test_start:test_end],
                )
            )
        return folds


def _fold_metrics_logreg(X_train_seq, y_train_seq, X_test_seq, y_test_seq) -> np.ndarray:
    clf = LogisticRegressionTabular(pooling="last").fit(X_train_seq, y_train_seq)
    return clf.predict_proba(X_test_seq)[:, 1]


def _fold_metrics_lstm(
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, fold_idx: int, epochs: int, seed: int
) -> np.ndarray:
    set_global_seed(seed)
    n_val = max(1, int(round(0.1 * len(X_train_seq))))
    X_tr, y_tr = X_train_seq[:-n_val], y_train_seq[:-n_val]
    X_val, y_val = X_train_seq[-n_val:], y_train_seq[-n_val:]

    train_loader, val_loader, test_loader = make_loaders(
        X_tr, y_tr, X_val, y_val, X_test_seq, y_test_seq, batch_size=128
    )
    model = LSTMClassifier(input_dim=X_train_seq.shape[2], hidden_size=64, num_layers=1, dropout=0.1)

    tmp_dir = Path("data/artifacts/walkforward_tmp")
    ensure_dir(str(tmp_dir))
    cfg_train = TrainConfig(
        lr=1e-3,
        epochs=epochs,
        patience=3,
        monitor="pr_auc",
        ckpt_path=str(tmp_dir / f"fold_{fold_idx}.pt"),
        log_csv=str(tmp_dir / f"fold_{fold_idx}_logs.csv"),
    )
    fit(model, train_loader, val_loader, cfg_train)

    device = get_device()
    model.to(device)
    model.eval()

    state = torch.load(cfg_train.ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    all_logits = []
    with torch.no_grad():
        for X, _y in test_loader:
            all_logits.append(model(X.to(device)))
    logits = torch.cat(all_logits).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def run_walkforward(
    cfg: DataConfig,
    seq_len: int = 64,
    n_splits: int = 5,
    test_size: int = 90,
    min_train_size: int = 500,
    model: str = "logreg",
    lstm_epochs: int = 8,
    seed: int = 42,
    fees_bps: float = 10.0,
    theta: float = 0.5,
) -> dict:
    """Run walk-forward CV and return a JSON-serializable report dict.

    theta is held fixed (not tuned per fold) on purpose: tuning a threshold
    on each fold's own test set would be leakage, and tuning it on a val
    split we don't otherwise carve out here would add another moving part.
    A fixed theta=0.5 keeps every fold's evaluation honestly out-of-sample.
    """
    if model not in {"logreg", "lstm"}:
        raise ValueError("model must be 'logreg' or 'lstm'")

    data = build_feature_frame(cfg)
    features = [c for c in FEATURE_COLUMNS_DEFAULT if c in data.columns]
    close = data["Close"]

    splitter = WalkForwardSplitter(
        n_splits=n_splits, min_train_size=min_train_size, test_size=test_size, embargo=cfg.horizon
    )
    folds = splitter.split(data)
    if not folds:
        raise ValueError(
            f"No fold satisfies min_train_size={min_train_size} with only {len(data)} rows available; "
            "reduce min_train_size/test_size/n_splits or widen the date range."
        )

    per_fold = []
    for fold in folds:
        X_train_df = data.loc[fold.train_idx, features]
        X_test_df = data.loc[fold.test_idx, features]
        y_train = data.loc[fold.train_idx, "target"].values
        y_test = data.loc[fold.test_idx, "target"].values

        scaler = fit_scaler(X_train_df, robust=cfg.use_robust_scaler)
        X_train = transform_with_scaler(scaler, X_train_df)
        X_test = transform_with_scaler(scaler, X_test_df)

        X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len=seq_len)
        X_test_seq, y_test_seq = build_sequences(X_test, y_test, seq_len=seq_len)

        if len(X_train_seq) < 10 or len(X_test_seq) < 5:
            continue  # fold too small to be meaningful once windowed

        if model == "logreg":
            proba = _fold_metrics_logreg(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
        else:
            proba = _fold_metrics_lstm(
                X_train_seq, y_train_seq, X_test_seq, y_test_seq, fold.index, lstm_epochs, seed
            )

        roc = float(roc_auc_score(y_test_seq, proba)) if len(np.unique(y_test_seq)) > 1 else float("nan")
        pr = float(average_precision_score(y_test_seq, proba))

        # Sequence target i corresponds to row (seq_len + i) of the fold's test slice.
        test_target_idx = fold.test_idx[seq_len:]
        proba_s = pd.Series(proba, index=test_target_idx)
        ret_asset = close.pct_change().reindex(test_target_idx)
        signal = signal_from_proba(proba_s, theta=theta, long_short=False)
        bt = backtest(ret_asset, signal, fees_bps=fees_bps)
        kpis = summary_kpis(bt.df)

        per_fold.append(
            {
                "fold": fold.index,
                "train_start": str(fold.train_idx.min()),
                "train_end": str(fold.train_idx.max()),
                "test_start": str(fold.test_idx.min()),
                "test_end": str(fold.test_idx.max()),
                "n_train": int(len(X_train_seq)),
                "n_test": int(len(X_test_seq)),
                "roc_auc": roc,
                "pr_auc": pr,
                "sharpe": float(kpis["Sharpe"]) if not np.isnan(kpis["Sharpe"]) else None,
                "cagr": float(kpis["CAGR"]),
                "max_drawdown": float(kpis["MaxDrawdown"]),
            }
        )

    if not per_fold:
        raise ValueError("Every fold was too small once windowed by seq_len; widen the date range.")

    roc_values = [f["roc_auc"] for f in per_fold if not np.isnan(f["roc_auc"])]
    pr_values = [f["pr_auc"] for f in per_fold]
    sharpe_values = [f["sharpe"] for f in per_fold if f["sharpe"] is not None]

    report = {
        "meta": {
            "ticker": cfg.ticker,
            "model": model,
            "n_splits_requested": n_splits,
            "n_folds_run": len(per_fold),
            "test_size": test_size,
            "min_train_size": min_train_size,
            "embargo": cfg.horizon,
            "theta": theta,
            "fees_bps": fees_bps,
        },
        "folds": per_fold,
        "summary": {
            "roc_auc_mean": float(np.mean(roc_values)) if roc_values else None,
            "roc_auc_std": float(np.std(roc_values, ddof=1)) if len(roc_values) > 1 else None,
            "pr_auc_mean": float(np.mean(pr_values)) if pr_values else None,
            "pr_auc_std": float(np.std(pr_values, ddof=1)) if len(pr_values) > 1 else None,
            "sharpe_mean": float(np.mean(sharpe_values)) if sharpe_values else None,
            "sharpe_std": float(np.std(sharpe_values, ddof=1)) if len(sharpe_values) > 1 else None,
        },
    }
    return report


def plot_walkforward_report(report: dict, outpath: str) -> None:
    folds = report["folds"]
    x = [f["fold"] for f in folds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(x, [f["roc_auc"] for f in folds], marker="o", label="ROC-AUC")
    axes[0].plot(x, [f["pr_auc"] for f in folds], marker="o", label="PR-AUC")
    axes[0].axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("Fold")
    axes[0].set_title("Out-of-sample discrimination per fold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sharpes = [f["sharpe"] if f["sharpe"] is not None else np.nan for f in folds]
    axes[1].bar(x, sharpes)
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].set_xlabel("Fold")
    axes[1].set_title(f"Backtest Sharpe per fold (θ={report['meta']['theta']:.2f})")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Walk-forward (purged, expanding-window) cross-validation")
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=int, default=90)
    p.add_argument("--min-train-size", type=int, default=500)
    p.add_argument("--model", type=str, default="logreg", choices=["logreg", "lstm"])
    p.add_argument("--lstm-epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--out", type=str, default="data/artifacts/walkforward_report.json")
    p.add_argument("--fig", type=str, default="experiments/figures/walkforward_metrics.png")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        horizon=args.horizon,
        test_start=None,  # walk-forward owns its own splitting; not used here
    )
    report = run_walkforward(
        cfg,
        seq_len=args.seq_len,
        n_splits=args.n_splits,
        test_size=args.test_size,
        min_train_size=args.min_train_size,
        model=args.model,
        lstm_epochs=args.lstm_epochs,
        seed=args.seed,
        fees_bps=args.fees_bps,
        theta=args.theta,
    )

    ensure_dir(str(Path(args.out).parent))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved walk-forward report -> {args.out}")
    print(json.dumps(report["summary"], indent=2))

    ensure_dir(str(Path(args.fig).parent))
    plot_walkforward_report(report, args.fig)
    print(f"Saved plot -> {args.fig}")


if __name__ == "__main__":
    main()
