# -----------------------------
# File: src/validation/multiseed.py
# -----------------------------
"""Multi-seed training aggregation with confidence intervals.

A single training run's test-set metric is one draw from a noisy process:
weight initialization, minibatch shuffling order, and dropout masks all
differ by seed. Reporting a single number (as the base pipeline does) invites
the reader to treat e.g. "ROC-AUC=0.516" as more precise than it is. This
module retrains the same architecture across several seeds on the exact same
data split and reports mean +/- a confidence interval, so the report is
honest about how much of the headline number is signal vs. training noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.config import DataConfig
from ..data.dataset import prepare_dataset
from ..models.lstm import LSTMClassifier
from ..training.dataloaders import make_loaders
from ..training.trainer import TrainConfig, fit
from ..training.utils import ensure_dir, get_device, set_global_seed


@dataclass
class SeedResult:
    seed: int
    best_epoch: int
    best_val_score: float
    test_roc_auc: float
    test_pr_auc: float


def _mean_ci(values: list[float], confidence: float = 0.95) -> dict:
    """Mean, sample std, and a t-distribution confidence interval.

    Uses Student's t (not a normal approximation) because n is always small
    here (a handful of seeds) -- the t-distribution's fatter tails are the
    honest choice at small sample sizes.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "std": None, "ci_low": None, "ci_high": None, "n": n}
    std = float(arr.std(ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "n": n,
        "confidence": confidence,
    }


def run_multiseed(
    cfg: DataConfig,
    seq_len: int = 64,
    seeds: list[int] = (1, 2, 3, 4, 5),
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 30,
    patience: int = 5,
    batch: int = 256,
    lr: float = 1e-3,
    tmp_dir: str = "data/artifacts/multiseed_tmp",
) -> dict:
    # Data doesn't depend on the seed: build it once and reuse across seeds,
    # so all seeds are compared on the exact same train/val/test split.
    data = prepare_dataset(cfg, seq_len=seq_len)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_features = X_train.shape[2]

    ensure_dir(tmp_dir)
    device = get_device()

    results: list[SeedResult] = []
    for seed in seeds:
        set_global_seed(seed)
        train_loader, val_loader, test_loader = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch
        )
        model = LSTMClassifier(input_dim=n_features, hidden_size=hidden, num_layers=layers, dropout=dropout)
        cfg_train = TrainConfig(
            lr=lr,
            epochs=epochs,
            patience=patience,
            ckpt_path=str(Path(tmp_dir) / f"seed_{seed}.pt"),
            log_csv=str(Path(tmp_dir) / f"seed_{seed}_logs.csv"),
        )
        summary = fit(model, train_loader, val_loader, cfg_train)

        state = torch.load(cfg_train.ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()

        all_logits, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                all_logits.append(model(X.to(device)))
                all_targets.append(y)
        logits = torch.cat(all_logits).cpu().numpy()
        y_true = torch.cat(all_targets).numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))

        results.append(
            SeedResult(
                seed=seed,
                best_epoch=int(summary["best_epoch"]),
                best_val_score=float(summary["best_score"]),
                test_roc_auc=float(roc_auc_score(y_true, proba)),
                test_pr_auc=float(average_precision_score(y_true, proba)),
            )
        )

    roc_values = [r.test_roc_auc for r in results]
    pr_values = [r.test_pr_auc for r in results]

    report = {
        "meta": {
            "ticker": cfg.ticker,
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "epochs_budget": epochs,
        },
        "per_seed": [r.__dict__ for r in results],
        "summary": {
            "test_roc_auc": _mean_ci(roc_values),
            "test_pr_auc": _mean_ci(pr_values),
        },
    }
    return report


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Multi-seed LSTM training with confidence intervals")
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="data/artifacts/multiseed_report.json")
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
    report = run_multiseed(
        cfg,
        seq_len=args.seq_len,
        seeds=args.seeds,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        lr=args.lr,
    )
    ensure_dir(str(Path(args.out).parent))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved multi-seed report -> {args.out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
