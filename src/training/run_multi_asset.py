# -----------------------------
# File: src/training/run_multi_asset.py
# -----------------------------
"""Train the same LSTM architecture across several assets and compare.

Demonstrates the pipeline isn't hand-fit to BTC-USD: DataConfig.ticker was
already generic, so this reuses train_lstm() (src/training/run_lstm.py) --
the exact same training path as `make run` -- in-process for each ticker
(no subprocess), so failures on one asset don't require re-plumbing CLI
flags and results are trivially aggregable in memory.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..data.config import DataConfig
from ..data.paths import artifact_prefix
from ..training.utils import ensure_dir
from .run_lstm import train_lstm

DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "^GSPC"]


def _artifact_prefix(ticker: str) -> str:
    return artifact_prefix("lstm", ticker)


def run_multi_asset(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2018-01-01",
    test_start: str = "2023-01-01",
    seq_len: int = 64,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 1337,
    artifacts_dir: str = "data/artifacts",
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        cfg = DataConfig(ticker=ticker, start=start, test_start=test_start)
        prefix = _artifact_prefix(ticker)
        try:
            report = train_lstm(
                cfg,
                seq_len=seq_len,
                epochs=epochs,
                patience=patience,
                seed=seed,
                artifacts_dir=artifacts_dir,
                artifact_prefix=prefix,
            )
            rows.append(
                {
                    "ticker": ticker,
                    "best_epoch": report["train_summary"]["best_epoch"],
                    "test_roc_auc": report["test_roc_auc"],
                    "test_pr_auc": report["test_pr_auc"],
                    "error": None,
                }
            )
        except Exception as exc:  # one bad asset shouldn't kill the whole sweep
            print(f"[run_multi_asset] {ticker} failed:\n{traceback.format_exc()}")
            rows.append(
                {
                    "ticker": ticker,
                    "best_epoch": None,
                    "test_roc_auc": None,
                    "test_pr_auc": None,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def plot_multi_asset_comparison(df: pd.DataFrame, outpath: str) -> None:
    ok = df.dropna(subset=["test_roc_auc"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ok["ticker"], ok["test_roc_auc"])
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Random (0.50)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("LSTM test ROC-AUC across assets")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Train the LSTM across multiple assets and compare")
    p.add_argument("--tickers", type=str, nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="data/artifacts/multi_asset_comparison.csv")
    p.add_argument("--fig", type=str, default="experiments/figures/multi_asset_comparison.png")
    return p.parse_args()


def main():
    args = _parse_args()
    df = run_multi_asset(
        tickers=args.tickers,
        start=args.start,
        test_start=args.test_start,
        seq_len=args.seq_len,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    ensure_dir(str(Path(args.out).parent))
    df.to_csv(args.out, index=False)
    print(f"Saved comparison -> {args.out}")
    print(df)

    ensure_dir(str(Path(args.fig).parent))
    plot_multi_asset_comparison(df, args.fig)
    print(f"Saved plot -> {args.fig}")


if __name__ == "__main__":
    main()
