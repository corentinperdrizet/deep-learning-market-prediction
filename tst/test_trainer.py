from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.lstm import LSTMClassifier
from src.training.dataloaders import make_loaders
from src.training.trainer import TrainConfig, _make_optimizer, fit


def test_make_optimizer_adam_and_adamw():
    params = [torch.nn.Parameter(torch.zeros(2))]
    assert isinstance(_make_optimizer("adam", params, 1e-3, 0.0), torch.optim.Adam)
    assert isinstance(_make_optimizer("adamw", params, 1e-3, 0.0), torch.optim.AdamW)


def test_make_optimizer_rejects_unknown_name():
    params = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(ValueError):
        _make_optimizer("sgd", params, 1e-3, 0.0)


def test_fit_persists_model_config_alongside_checkpoint(tmp_path):
    """Regression guard for the serving layer (src/serving/): fit() must save
    enough architecture metadata that a checkpoint is self-describing, not
    just its TrainConfig (training hyperparameters, a different thing).
    """
    rng = np.random.default_rng(0)
    n, seq_len, n_features = 80, 5, 3
    X = rng.normal(size=(n, seq_len, n_features)).astype("float32")
    y = (rng.uniform(size=n) > 0.5).astype("float32")

    train_loader, val_loader, _ = make_loaders(X[:60], y[:60], X[60:], y[60:], batch_size=16)

    model_config = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "bidirectional": False}
    model = LSTMClassifier(input_dim=n_features, **model_config)

    cfg = TrainConfig(
        epochs=2,
        patience=2,
        ckpt_path=str(tmp_path / "model.pt"),
        log_csv=str(tmp_path / "logs.csv"),
        optimizer="adamw",
    )
    fit(model, train_loader, val_loader, cfg, model_config=model_config)

    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    assert checkpoint["model_config"] == model_config
    assert checkpoint["config"]["optimizer"] == "adamw"
    assert "model_state" in checkpoint
