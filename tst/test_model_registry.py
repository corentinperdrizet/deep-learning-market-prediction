from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

import src.serving.model_registry as reg
from src.data.scaling import fit_scaler
from src.models.lstm import LSTMClassifier


def _write_lstm_checkpoint(path, model_config, seed=0):
    torch.manual_seed(seed)
    model = LSTMClassifier(
        input_dim=model_config["input_dim"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        bidirectional=model_config["bidirectional"],
    )
    torch.save(
        {"epoch": 1, "model_state": model.state_dict(), "config": {}, "model_config": model_config}, path
    )
    return model


def test_list_available_models_parses_prefixes(tmp_path):
    for name in ["lstm_classifier.pt", "transformer_classifier.pt", "lstm_ETH-USD_classifier.pt"]:
        (tmp_path / name).write_bytes(b"not a real checkpoint, just needs to exist")

    found = reg.list_available_models(str(tmp_path))
    pairs = {(m["model_kind"], m["ticker"]) for m in found}
    assert pairs == {("lstm", "BTC-USD"), ("transformer", "BTC-USD"), ("lstm", "ETH-USD")}


def test_list_available_models_empty_dir_returns_empty_list(tmp_path):
    assert reg.list_available_models(str(tmp_path / "does_not_exist")) == []


def test_load_model_reconstructs_architecture_from_model_config(tmp_path):
    model_config = {
        "input_dim": 5,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "seq_len": 8,
    }
    ckpt_path = tmp_path / "lstm_classifier.pt"
    reference = _write_lstm_checkpoint(ckpt_path, model_config)

    model, loaded_config, device = reg.load_model("lstm", "BTC-USD", artifacts_dir=str(tmp_path))

    assert loaded_config == model_config
    x = torch.randn(2, model_config["seq_len"], model_config["input_dim"])
    with torch.no_grad():
        out_loaded = model(x.to(device)).cpu()
        out_reference = reference(x)
    assert torch.allclose(out_loaded, out_reference, atol=1e-5)


def test_load_model_raises_not_found_for_missing_checkpoint(tmp_path):
    with pytest.raises(reg.ModelNotFoundError):
        reg.load_model("lstm", "BTC-USD", artifacts_dir=str(tmp_path))


def test_load_model_raises_value_error_when_model_config_missing(tmp_path):
    ckpt_path = tmp_path / "lstm_classifier.pt"
    torch.save({"epoch": 1, "model_state": {}, "config": {}, "model_config": None}, ckpt_path)
    with pytest.raises(ValueError, match="model_config"):
        reg.load_model("lstm", "BTC-USD", artifacts_dir=str(tmp_path))


def test_load_model_rejects_unknown_model_kind(tmp_path):
    with pytest.raises(ValueError, match="Unknown model_kind"):
        reg.load_model("xgboost", "BTC-USD", artifacts_dir=str(tmp_path))


def test_build_latest_window_shape_and_scaling(monkeypatch, tmp_path):
    seq_len = 10
    n_features = 3
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    fake_frame = pd.DataFrame(
        {
            "log_ret": np.linspace(-1, 1, 50),
            "vol_20": np.linspace(0, 1, 50),
            "rsi_14": np.linspace(0, 100, 50),
        },
        index=idx,
    )
    monkeypatch.setattr(reg, "build_feature_frame", lambda cfg: fake_frame)
    monkeypatch.setattr(reg, "FEATURE_COLUMNS_DEFAULT", ["log_ret", "vol_20", "rsi_14", "extra_missing_col"])

    scaler = fit_scaler(fake_frame[["log_ret", "vol_20", "rsi_14"]])
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")

    X, as_of = reg.build_latest_window("BTC-USD", seq_len, artifacts_dir=str(artifacts_dir))

    assert X.shape == (1, seq_len, n_features)
    assert as_of == str(fake_frame.index[-1])


def test_build_latest_window_raises_when_not_enough_history(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    fake_frame = pd.DataFrame({"log_ret": [0.0] * 5}, index=idx)
    monkeypatch.setattr(reg, "build_feature_frame", lambda cfg: fake_frame)
    with pytest.raises(ValueError, match="Not enough history"):
        reg.build_latest_window("BTC-USD", seq_len=64)


def test_load_threshold_defaults_to_half_when_missing(tmp_path):
    assert reg._load_threshold("lstm", str(tmp_path)) == 0.5


def test_load_threshold_reads_saved_value(tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps({"lstm": {"theta": 0.37}}))
    assert reg._load_threshold("lstm", str(tmp_path)) == pytest.approx(0.37)
