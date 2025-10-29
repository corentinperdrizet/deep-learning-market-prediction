# src/training/run_transformer.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==== Project imports (adaptés à ton repo) ====
from src.data.dataset import prepare_dataset, DataConfig
from src.models.transformer import TransformerTimeSeriesClassifier
from src.training.dataloaders import make_loaders
from src.training.utils import get_device  # ton util déjà existant (CUDA -> MPS -> CPU)

# Essaye d'utiliser tes helpers si disponibles; sinon fallback interne.
try:
    from src.training.trainer import Trainer as ProjectTrainer  # type: ignore
except Exception:
    ProjectTrainer = None  # fallback ci-dessous

try:
    from src.training.evaluate import evaluate_classifier as project_evaluate_classifier  # type: ignore
except Exception:
    project_evaluate_classifier = None


def set_seed(seed: int = 42):
    import random
    import numpy as _np
    torch.manual_seed(seed)
    random.seed(seed)
    _np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_two_column_proba(p1: np.ndarray) -> np.ndarray:
    """
    Convertit un vecteur de proba P(class=1) -> matrice [P0, P1] attendue par tes métriques.
    Clamp léger pour éviter 0/1 stricts.
    """
    p1 = np.asarray(p1).reshape(-1)
    p1 = np.clip(p1, 1e-7, 1 - 1e-7)
    p0 = 1.0 - p1
    return np.column_stack([p0, p1])


def evaluate_classifier_fallback(y_true: np.ndarray,
                                 y_proba_1d: np.ndarray) -> Dict[str, float]:
    """
    Fallback simple si src/training/evaluate.py n'est pas compatible.
    Utilise scikit-learn si dispo; sinon renvoie un sous-ensemble minimal.
    """
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score, brier_score_loss
        )
        y_pred = (y_proba_1d >= 0.5).astype(int)
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba_1d))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba_1d))
        out["f1"] = float(f1_score(y_true, y_pred))
        out["brier"] = float(brier_score_loss(y_true, y_proba_1d))
    except Exception:
        # Minimal si sklearn absent
        y_pred = (y_proba_1d >= 0.5).astype(int)
        acc = float((y_pred == y_true).mean())
        out = {"accuracy": acc}
    return out


class TrainerLite:
    """
    Entraîneur minimal (fallback) compatible BCEWithLogits pour classification binaire.
    Loggue: epoch, train_loss, val_loss, pr_auc, roc_auc, f1, lr
    Early stopping sur pr_auc.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        artifacts_dir: str,
        logs_name: str,
        ckpt_name: str,
        early_stopping_metric: str = "pr_auc",
        mode: str = "max",
        patience: int = 5,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        self.artifacts_dir = Path(artifacts_dir)
        self.logs_path = self.artifacts_dir / logs_name
        self.ckpt_path = self.artifacts_dir / ckpt_name

        self.es_metric = early_stopping_metric
        self.mode = mode
        self.patience = patience
        self.best = -float("inf") if mode == "max" else float("inf")
        self.wait = 0

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # init CSV
        with open(self.logs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "pr_auc", "roc_auc", "f1", "lr"])

    def _step(self, batch, train: bool):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return loss.item(), logits.detach().cpu(), y.detach().cpu()
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                loss = self.criterion(logits, y)
            return loss.item(), logits.detach().cpu(), y.detach().cpu()

    def _gather_epoch(self, loader, train: bool):
        losses = []
        all_logits, all_y = [], []
        for batch in loader:
            loss, logits, y = self._step(batch, train=train)
            losses.append(loss)
            all_logits.append(logits)
            all_y.append(y)
        loss_avg = float(np.mean(losses)) if losses else 0.0
        logits = torch.cat(all_logits).numpy()
        y = torch.cat(all_y).numpy()
        proba = 1 / (1 + np.exp(-logits))  # sigmoid
        return loss_avg, y, proba

    def fit(self, train_loader, val_loader, epochs: int):
        for epoch in range(1, epochs + 1):
            train_loss, _, _ = self._gather_epoch(train_loader, train=True)
            val_loss, y_val, p_val = self._gather_epoch(val_loader, train=False)

            # metrics
            if project_evaluate_classifier is not None:
                p_val_2d = _to_two_column_proba(p_val)
                metrics = project_evaluate_classifier(y_val, p_val_2d)
            else:
                metrics = evaluate_classifier_fallback(y_val, p_val)

            pr_auc = float(metrics.get("pr_auc", float("nan")))
            roc_auc = float(metrics.get("roc_auc", float("nan")))
            f1 = float(metrics.get("f1", float("nan")))

            # scheduler (ReduceLROnPlateau accepté)
            if self.scheduler is not None:
                try:
                    self.scheduler.step(pr_auc)
                except Exception:
                    pass

            # log CSV
            lr_current = self.optimizer.param_groups[0]["lr"]
            with open(self.logs_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, pr_auc, roc_auc, f1, lr_current])

            # early stopping
            score = metrics.get(self.es_metric, None)
            improved = False
            if score is not None:
                if self.mode == "max" and score > self.best:
                    improved = True
                elif self.mode == "min" and score < self.best:
                    improved = True
            if improved:
                self.best = score
                self.wait = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    break

    def evaluate(self, loader):
        _, y, p = self._gather_epoch(loader, train=False)
        if project_evaluate_classifier is not None:
            p2d = _to_two_column_proba(p)
            return project_evaluate_classifier(y, p2d)
        return evaluate_classifier_fallback(y, p)


def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer Time-Series Classifier")

    # Data
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--label-type", type=str, default="direction", choices=["direction", "return"])
    p.add_argument("--seq-len", type=int, default=64)

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])

    # Training
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--patience", type=int, default=6)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--artifacts-dir", type=str, default="data/artifacts")
    p.add_argument("--logs-name", type=str, default="transformer_logs.csv")
    p.add_argument("--ckpt-name", type=str, default="transformer_classifier.pt")
    p.add_argument("--test-report", type=str, default="transformer_test_report.json")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()  # ton util existant
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # === Dataset ===
    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        test_start=args.test_start,
        horizon=args.horizon,
        label_type=args.label_type,
    )
    data = prepare_dataset(cfg, seq_len=args.seq_len)
    features = data["features"]
    input_dim = len(features)

    # === Dataloaders (note: ton make_loaders renvoie un tuple) ===
    train_loader, val_loader, test_loader = make_loaders(
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"], y_val=data["y_val"],
        X_test=data["X_test"], y_test=data["y_test"],
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # === Model / Loss / Optim ===
    model = TransformerTimeSeriesClassifier(
        input_dim=input_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        pooling=args.pooling,
        max_len=max(4096, args.seq_len + 10),
        layer_norm_final=True,
    ).to(device)

    if args.pos_weight is not None:
        pos_w = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    else:
        pos_w = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler optionnel: ReduceLROnPlateau sur pr_auc
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    except Exception:
        scheduler = None

    # === Trainer: on privilégie ton Trainer s'il existe, sinon fallback ===
    if ProjectTrainer is not None:
        trainer = ProjectTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            artifacts_dir=str(artifacts_dir),
            logs_name=args.logs_name,
            ckpt_name=args.ckpt_name,
            early_stopping_metric="pr_auc",
            mode="max",
        )
        trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)
        test_metrics = trainer.evaluate(test_loader)
    else:
        trainer = TrainerLite(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            artifacts_dir=str(artifacts_dir),
            logs_name=args.logs_name,
            ckpt_name=args.ckpt_name,
            early_stopping_metric="pr_auc",
            mode="max",
            patience=args.patience,
            scheduler=scheduler,
        )
        trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)
        test_metrics = trainer.evaluate(test_loader)

    # === Rapport test ===
    report_path = artifacts_dir / args.test_report
    with open(report_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[OK] Test metrics written to: {report_path}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
