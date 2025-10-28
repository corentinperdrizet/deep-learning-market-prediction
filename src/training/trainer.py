# src/training/trainer.py
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import get_device, save_checkpoint, ensure_dir


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 5
    monitor: str = "pr_auc"  # "pr_auc" or "f1" or "roc_auc"
    ckpt_path: str = "data/artifacts/lstm_classifier.pt"
    log_csv: str = "data/artifacts/lstm_logs.csv"
    weight_decay: float = 0.0
    pos_weight: Optional[float] = None          # for class imbalance: BCEWithLogitsLoss(pos_weight=...)
    scheduler_reduce_lr: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6
    grad_clip_norm: Optional[float] = 1.0


def _compute_metrics(y_true: torch.Tensor, y_logits: torch.Tensor) -> Dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy()
    y_proba_np = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_pred_np = (y_proba_np >= 0.5).astype("int32")

    metrics = {}
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_np, y_proba_np))
    except Exception:
        metrics["pr_auc"] = float("nan")
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_proba_np))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["f1"] = float(f1_score(y_true_np, y_pred_np))
    except Exception:
        metrics["f1"] = float("nan")
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_logits = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        bs = y.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        all_logits.append(logits)
        all_targets.append(y)

    val_loss = running_loss / max(n_samples, 1)
    y_logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    metrics = _compute_metrics(y_true, y_logits)
    return val_loss, metrics


def _metric_value(metrics: Dict[str, float], name: str) -> float:
    return metrics.get(name, float("nan"))


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, float]:
    device = get_device()
    model = model.to(device)

    pos_weight_tensor = None
    if cfg.pos_weight is not None:
        pos_weight_tensor = torch.tensor([cfg.pos_weight], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Compat: certaines versions de torch n'acceptent pas 'verbose' dans ReduceLROnPlateau
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
        if cfg.scheduler_reduce_lr
        else None
    )

    ensure_dir(os.path.dirname(cfg.ckpt_path))
    ensure_dir(os.path.dirname(cfg.log_csv))

    with open(cfg.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_pr_auc",
                "val_roc_auc",
                "val_f1",
                "lr",
            ],
        )
        writer.writeheader()

    best_score = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_norm=cfg.grad_clip_norm
        )
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        monitor_value = _metric_value(val_metrics, cfg.monitor)
        if scheduler is not None:
            scheduler.step(monitor_value)

        current_lr = optimizer.param_groups[0]["lr"]

        with open(cfg.log_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_pr_auc",
                    "val_roc_auc",
                    "val_f1",
                    "lr",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_pr_auc": f"{val_metrics.get('pr_auc', float('nan')):.6f}",
                    "val_roc_auc": f"{val_metrics.get('roc_auc', float('nan')):.6f}",
                    "val_f1": f"{val_metrics.get('f1', float('nan')):.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )

        improved = monitor_value > best_score
        if improved:
            best_score = monitor_value
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "config": asdict(cfg)}, cfg.ckpt_path)
        else:
            epochs_no_improve += 1

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"PR-AUC={val_metrics.get('pr_auc', float('nan')):.4f} "
            f"ROC-AUC={val_metrics.get('roc_auc', float('nan')):.4f} "
            f"F1={val_metrics.get('f1', float('nan')):.4f} "
            f"lr={current_lr:.6f}"
        )

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (best {cfg.monitor}={best_score:.4f} at epoch {best_epoch})")
            break

    return {"best_epoch": best_epoch, "best_score": best_score, "monitor": cfg.monitor}
