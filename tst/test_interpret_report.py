from __future__ import annotations

import numpy as np
import torch

import src.interpret.report as report_mod
from src.data.config import DataConfig
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerTimeSeriesClassifier


def _fake_dataset(n_test=40, seq_len=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X_test = rng.normal(size=(n_test, seq_len, n_features)).astype("float32")
    y_test = rng.integers(0, 2, size=n_test)
    return {
        "X_train": X_test,
        "y_train": y_test,
        "X_val": X_test,
        "y_val": y_test,
        "X_test": X_test,
        "y_test": y_test,
        "features": ["f0", "f1", "f2"],
        "idx": {},
        "meta": {},
    }


def test_report_generates_importance_artifacts_for_lstm(monkeypatch, tmp_path):
    fake_data = _fake_dataset()
    monkeypatch.setattr(report_mod, "prepare_dataset", lambda cfg, seq_len: fake_data)

    model = LSTMClassifier(input_dim=3, hidden_size=4, num_layers=1)
    model.eval()
    monkeypatch.setattr(
        report_mod,
        "load_model",
        lambda kind, ticker, artifacts_dir: (model, {"seq_len": 6}, torch.device("cpu")),
    )

    result = report_mod.run_interpretability_report(
        DataConfig(ticker="BTC-USD"),
        model_kind="lstm",
        n_repeats=2,
        artifacts_dir=str(tmp_path / "artifacts"),
        figures_dir=str(tmp_path / "figures"),
    )

    assert (tmp_path / "artifacts" / "lstm_feature_importance.csv").exists()
    assert (tmp_path / "figures" / "lstm_feature_importance.png").exists()
    assert "attention_fig" not in result
    assert 0.0 <= result["baseline_roc_auc"] <= 1.0


def test_report_also_generates_attention_artifact_for_transformer(monkeypatch, tmp_path):
    fake_data = _fake_dataset()
    monkeypatch.setattr(report_mod, "prepare_dataset", lambda cfg, seq_len: fake_data)

    model = TransformerTimeSeriesClassifier(input_dim=3, d_model=8, n_heads=2, n_layers=1, pooling="mean")
    model.eval()
    monkeypatch.setattr(
        report_mod,
        "load_model",
        lambda kind, ticker, artifacts_dir: (model, {"seq_len": 6, "pooling": "mean"}, torch.device("cpu")),
    )

    result = report_mod.run_interpretability_report(
        DataConfig(ticker="BTC-USD"),
        model_kind="transformer",
        n_repeats=2,
        artifacts_dir=str(tmp_path / "artifacts"),
        figures_dir=str(tmp_path / "figures"),
    )

    assert (tmp_path / "figures" / "transformer_attention.png").exists()
    assert result["attention_fig"] == str(tmp_path / "figures" / "transformer_attention.png")
