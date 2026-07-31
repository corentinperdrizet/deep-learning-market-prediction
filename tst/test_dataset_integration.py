"""End-to-end integration test of the data pipeline (loaders -> quality ->
features -> preprocessing -> scaling -> sequences -> dataset), with network
access fully mocked. This is the single most valuable test in the suite:
it exercises the exact code path `python -m src.data.dataset` runs, and
would catch a broken import, a shape mismatch, or a reintroduced leakage
bug across module boundaries -- the kind of failure per-module unit tests
can miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data.dataset as dataset_mod
import src.data.loaders as loaders_mod
import src.data.scaling as scaling_mod
from src.data.config import DataConfig


def _fake_ohlcv(n=2400, start="2018-01-01", seed=42):
    # n=2400 daily bars from 2018-01-01 reaches mid-2024, comfortably past the
    # default test_start="2023-01-01" used below so the test split is non-empty.
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    log_rets = rng.normal(0.0003, 0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1000, 5000, size=n).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def sandboxed_pipeline(monkeypatch, tmp_path):
    """Redirect every module's cached `data_dir` reference to a tmp dir and
    stub out the network call, so the test never touches the real data/
    directory or the internet.
    """
    for mod in (loaders_mod, dataset_mod, scaling_mod):
        monkeypatch.setattr(mod, "data_dir", lambda: tmp_path)
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    fake_df = _fake_ohlcv()

    def fake_download(ticker, start, end, interval, auto_adjust, group_by, progress):
        return fake_df.copy()

    monkeypatch.setattr(loaders_mod.yf, "download", fake_download)
    return tmp_path


def test_prepare_dataset_end_to_end_shapes_and_no_leakage(sandboxed_pipeline):
    cfg = DataConfig(
        ticker="BTC-USD",
        start="2018-01-01",
        test_start="2023-01-01",
        cache_raw=True,
        cache_processed=True,
    )
    seq_len = 64
    out = dataset_mod.prepare_dataset(cfg, seq_len=seq_len)

    for split in ("train", "val", "test"):
        X, y = out[f"X_{split}"], out[f"y_{split}"]
        assert len(X) > 0, f"{split} split is empty -- synthetic date range too short"
        assert X.ndim == 3
        assert X.shape[1] == seq_len
        assert X.shape[0] == y.shape[0]
        assert not np.isnan(X).any()

    n_features = len(out["features"])
    assert out["X_train"].shape[2] == n_features

    # Chronological ordering across splits, with no overlap.
    idx = out["idx"]
    if len(idx["train"]) and len(idx["val"]):
        assert idx["train"].max() < idx["val"].min()
    if len(idx["val"]) and len(idx["test"]):
        assert idx["val"].max() < idx["test"].min()
    assert (idx["test"] >= pd.Timestamp(cfg.test_start, tz="UTC")).all()

    # Labels are binary.
    assert set(np.unique(out["y_train"])) <= {0, 1}

    # The scaler artifact must have been persisted (fit on train only, see
    # test_scaling.py for the leakage guarantee itself).
    assert (sandboxed_pipeline / "artifacts" / "scaler.joblib").exists()


def test_prepare_dataset_caches_raw_download_across_calls(sandboxed_pipeline, monkeypatch):
    calls = {"n": 0}
    original = loaders_mod.yf.download

    def counting_download(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loaders_mod.yf, "download", counting_download)

    cfg = DataConfig(ticker="BTC-USD", start="2018-01-01", test_start="2023-01-01")
    dataset_mod.prepare_dataset(cfg, seq_len=64)
    dataset_mod.prepare_dataset(cfg, seq_len=64)

    assert calls["n"] == 1
