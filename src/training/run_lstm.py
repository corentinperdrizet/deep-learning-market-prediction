# src/training/run_lstm.py
"""
Train and evaluate the first DL model (LSTM) end-to-end.

Usage (defaults: BTC-USD, 1d, start=2018-01-01, test_start=2023-01-01):
    python -m src.training.run_lstm
You can override:
    python -m src.training.run_lstm --ticker ETH-USD --interval 1d --start 2019-01-01 --test-start 2024-01-01 --horizon 1 --seq-len 64
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.config import DataConfig
from ..data.dataset import prepare_dataset
from ..data.paths import artifact_prefix as default_artifact_prefix
from ..models.lstm import LSTMClassifier
from .dataloaders import make_loaders
from .trainer import TrainConfig, fit
from .utils import get_device, set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run LSTM classifier on prepared dataset")
    # Data
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--val-start", type=str, default=None)
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--label-type", type=str, default="direction", choices=["direction", "return"])
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--robust-scaler", action="store_true", help="Use RobustScaler instead of StandardScaler")
    p.add_argument("--no-cache-raw", action="store_true")
    p.add_argument("--no-cache-processed", action="store_true")
    p.add_argument("--seq-len", type=int, default=64)

    # Model/Training
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")

    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--monitor", type=str, default="pr_auc", choices=["pr_auc", "roc_auc", "f1"])
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)

    # Artifacts
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/artifacts",
        help="Directory to write checkpoint/logs/report to",
    )
    p.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Prefix for artifact filenames (default: 'lstm' when ticker=BTC-USD for "
        "backward compatibility, else 'lstm_<ticker>' so multi-asset runs don't collide)",
    )

    return p.parse_args()


def train_lstm(
    cfg: DataConfig,
    seq_len: int = 64,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    batch: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    monitor: str = "pr_auc",
    pos_weight: float | None = None,
    grad_clip: float = 1.0,
    seed: int = 1337,
    artifacts_dir: str = "data/artifacts",
    artifact_prefix: str = "lstm",
) -> dict:
    """Prepare data, train an LSTMClassifier with early stopping, evaluate on
    test, and persist checkpoint/logs/report. Returns the JSON report dict.

    Factored out of main() so callers that need to train several models in
    one process (src/training/run_multi_asset.py, src/validation/*) can
    reuse the exact same training path as the CLI instead of shelling out.
    """
    set_global_seed(seed)

    data = prepare_dataset(cfg, seq_len=seq_len)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    features = data["features"]
    meta = data.get("meta", {})

    assert X_train.ndim == 3, "Expected (N, seq_len, n_features)"
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    print(
        f"[{cfg.ticker}] Dataset -> seq_len={seq_len}, n_features={n_features}, "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch
    )

    model_config = {
        "hidden_size": hidden,
        "num_layers": layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "input_dim": n_features,
        "seq_len": seq_len,
    }
    model = LSTMClassifier(
        input_dim=n_features,
        hidden_size=hidden,
        num_layers=layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    os.makedirs(artifacts_dir, exist_ok=True)
    ckpt_path = os.path.join(artifacts_dir, f"{artifact_prefix}_classifier.pt")
    log_csv = os.path.join(artifacts_dir, f"{artifact_prefix}_logs.csv")

    cfg_train = TrainConfig(
        lr=lr,
        epochs=epochs,
        patience=patience,
        monitor=monitor,
        ckpt_path=ckpt_path,
        log_csv=log_csv,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
        scheduler_reduce_lr=True,
        scheduler_factor=0.5,
        scheduler_patience=2,
        scheduler_min_lr=1e-6,
        grad_clip_norm=grad_clip,
        optimizer="adam",
    )

    summary = fit(model, train_loader, val_loader, cfg_train, model_config=model_config)
    print(f"[{cfg.ticker}] Train summary:", summary)

    device = get_device()
    state = torch.load(cfg_train.ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            all_logits.append(model(X))
            all_targets.append(y)

    y_logits = torch.cat(all_logits).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    y_proba = 1.0 / (1.0 + np.exp(-y_logits))

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc = float(roc_auc_score(y_true, y_proba))
    print(f"[{cfg.ticker}] [TEST] PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}")

    report = {
        "meta": meta,
        "features": features,
        "seq_len": int(seq_len),
        "seed": seed,
        "model_config": model_config,
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc,
        "train_summary": summary,
    }
    report_path = os.path.join(artifacts_dir, f"{artifact_prefix}_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{cfg.ticker}] Saved report -> {report_path}")

    return report


def main():
    args = parse_args()

    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        end=args.end,
        label_type=args.label_type,
        horizon=args.horizon,
        val_start=args.val_start,
        test_start=args.test_start,
        use_robust_scaler=args.robust_scaler,
        cache_raw=not args.no_cache_raw,
        cache_processed=not args.no_cache_processed,
    )

    prefix = args.artifact_prefix or default_artifact_prefix("lstm", args.ticker)

    train_lstm(
        cfg,
        seq_len=args.seq_len,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        monitor=args.monitor,
        pos_weight=args.pos_weight,
        grad_clip=args.grad_clip,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
        artifact_prefix=prefix,
    )


if __name__ == "__main__":
    main()
