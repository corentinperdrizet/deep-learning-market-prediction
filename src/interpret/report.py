# -----------------------------
# File: src/interpret/report.py
# -----------------------------
"""CLI driver: compute permutation feature importance (and, for the
Transformer, an attention map) for a trained model on its real test split,
and persist CSV/PNG artifacts the Streamlit dashboard's Interpretability tab
reads. Reuses src.serving.model_registry.load_model so a checkpoint is
loaded exactly the same way here as it is for serving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..data.config import DataConfig
from ..data.dataset import prepare_dataset
from ..data.paths import artifact_prefix
from ..serving.model_registry import load_model
from .attention import day_importance_from_attention, extract_attention_weights
from .importance import permutation_importance, torch_predict_fn


def run_interpretability_report(
    cfg: DataConfig,
    model_kind: str,
    seq_len: int = 64,
    n_repeats: int = 5,
    artifacts_dir: str = "data/artifacts",
    figures_dir: str = "experiments/figures",
) -> dict:
    data = prepare_dataset(cfg, seq_len=seq_len)
    X_test, y_test = data["X_test"], data["y_test"]
    features = data["features"]

    model, model_config, device = load_model(model_kind, cfg.ticker, artifacts_dir)
    prefix = artifact_prefix(model_kind, cfg.ticker)

    predict_fn = torch_predict_fn(model, device)
    imp_df = permutation_importance(predict_fn, X_test, y_test, feature_names=features, n_repeats=n_repeats)

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    imp_path = Path(artifacts_dir) / f"{prefix}_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = imp_df.iloc[::-1]
    ax.barh(ordered["feature"], ordered["importance_mean"])
    ax.set_xlabel("Importance (ROC-AUC drop when permuted)")
    ax.set_title(f"{model_kind.upper()} permutation feature importance ({cfg.ticker})")
    fig.tight_layout()
    imp_fig_path = Path(figures_dir) / f"{prefix}_feature_importance.png"
    fig.savefig(imp_fig_path, dpi=150)
    plt.close(fig)

    result = {
        "baseline_roc_auc": float(imp_df.attrs["baseline_score"]),
        "importance_csv": str(imp_path),
        "importance_fig": str(imp_fig_path),
    }

    if model_kind == "transformer":
        x_last = torch.from_numpy(X_test[-1:]).float().to(device)
        attn = extract_attention_weights(model, x_last)
        importance_days = day_importance_from_attention(attn, model_config["pooling"])[0]

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(range(len(importance_days)), importance_days)
        ax2.set_xlabel("Day offset in window (0 = oldest)")
        ax2.set_ylabel("Attention received")
        ax2.set_title(f"Attention over the last {len(importance_days)}-day window ({cfg.ticker})")
        fig2.tight_layout()
        attn_fig_path = Path(figures_dir) / f"{prefix}_attention.png"
        fig2.savefig(attn_fig_path, dpi=150)
        plt.close(fig2)
        result["attention_fig"] = str(attn_fig_path)

    return result


def _parse_args():
    p = argparse.ArgumentParser(description="Compute feature importance / attention for a trained model")
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    p.add_argument("--n-repeats", type=int, default=5)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        test_start=args.test_start,
        horizon=args.horizon,
    )
    result = run_interpretability_report(
        cfg, model_kind=args.model, seq_len=args.seq_len, n_repeats=args.n_repeats
    )
    print(result)


if __name__ == "__main__":
    main()
