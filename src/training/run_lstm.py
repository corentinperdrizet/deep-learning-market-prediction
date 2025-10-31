# src/training/run_lstm.py
"""
Train and evaluate the first DL model (LSTM) end-to-end.

Usage (defaults: BTC-USD, 1d, start=2018-01-01, test_start=2023-01-01):
    python -m src.training.run_lstm
You can override:
    python -m src.training.run_lstm --ticker ETH-USD --interval 1d --start 2019-01-01 --test-start 2024-01-01 --horizon 1 --seq-len 64
"""

import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.dataset import prepare_dataset
from ..data.config import DataConfig
from .dataloaders import make_loaders
from .utils import set_global_seed, get_device
from ..models.lstm import LSTMClassifier
from .trainer import TrainConfig, fit


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

    return p.parse_args()



def main():
    args = parse_args()
    set_global_seed(42)

    # ---- Build DataConfig expected by prepare_dataset ----
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
        # if your DataConfig has other fields, feel free to add them here
    )

    # ---- Prepare data (now passing cfg + seq_len) ----
    data = prepare_dataset(cfg, seq_len=args.seq_len)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    features = data["features"]
    meta = data.get("meta", {})

    assert X_train.ndim == 3, "Expected (N, seq_len, n_features)"
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    print(
        f"Dataset → seq_len={seq_len}, n_features={n_features}, "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # ---- Dataloaders ----
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch
    )

    # ---- Model ----
    model = LSTMClassifier(
        input_dim=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )

    # ---- Train config ----
    ckpt_dir = "data/artifacts"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "lstm_classifier.pt")
    log_csv = os.path.join(ckpt_dir, "lstm_logs.csv")

    cfg_train = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        monitor=args.monitor,
        ckpt_path=ckpt_path,
        log_csv=log_csv,
        weight_decay=args.weight_decay,
        pos_weight=(args.pos_weight if args.pos_weight is not None else None),
        scheduler_reduce_lr=True,
        scheduler_factor=0.5,
        scheduler_patience=2,
        scheduler_min_lr=1e-6,
        grad_clip_norm=args.grad_clip,
    )

    # ---- Fit ----
    summary = fit(model, train_loader, val_loader, cfg_train)
    print("Train summary:", summary)

    # ---- Load best checkpoint and evaluate on test ----
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
            logits = model(X)
            all_logits.append(logits)
            all_targets.append(y)

    y_logits = torch.cat(all_logits).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    y_proba = 1.0 / (1.0 + np.exp(-y_logits))

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc = float(roc_auc_score(y_true, y_proba))
    print(f"[TEST] PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}")

    # ---- Save JSON report ----
    report_path = os.path.join(ckpt_dir, "lstm_test_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "meta": meta,
                "features": features,
                "seq_len": int(seq_len),
                "test_pr_auc": pr_auc,
                "test_roc_auc": roc,
                "train_summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved report → {report_path}")


if __name__ == "__main__":
    main()
