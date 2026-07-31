from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.scaling import fit_scaler, load_scaler, save_scaler, transform_with_scaler


def test_scaler_is_fit_on_train_only_not_refit_on_val():
    """The whole no-leakage promise of the pipeline rests on this: the scaler
    must be fit once on train, then only ever *applied* (not refit) to val/test.
    We simulate this by fitting on a train set centered at 0 and transforming
    a val set centered far away -- if the scaler were refit on val, the
    transformed val mean would come back close to 0; since it's not refit,
    it must NOT be centered.
    """
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(500, 3)), columns=list("abc"))
    X_val = pd.DataFrame(rng.normal(loc=50.0, scale=1.0, size=(50, 3)), columns=list("abc"))

    scaler = fit_scaler(X_train, robust=False)
    X_val_scaled = transform_with_scaler(scaler, X_val)

    assert not np.allclose(X_val_scaled.mean(axis=0), 0.0, atol=1.0)
    # Train, by construction, should come back ~standardized.
    X_train_scaled = transform_with_scaler(scaler, X_train)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-8)


def test_transform_with_scaler_handles_empty_split():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scaler = fit_scaler(X_train)
    empty = pd.DataFrame({"a": [], "b": []})
    out = transform_with_scaler(scaler, empty)
    assert out.shape == (0, 2)


def test_save_and_load_scaler_are_ticker_scoped(monkeypatch, tmp_path):
    """Regression guard: two tickers must never share a scaler file --
    training on ETH-USD after BTC-USD must not silently corrupt BTC-USD's
    saved scaler (and therefore any model served against it).
    """
    import src.data.scaling as scaling_mod

    monkeypatch.setattr(scaling_mod, "data_dir", lambda: tmp_path)
    (tmp_path / "artifacts").mkdir(parents=True)

    btc_scaler = fit_scaler(pd.DataFrame({"a": [0.0, 1.0, 2.0]}))
    eth_scaler = fit_scaler(pd.DataFrame({"a": [100.0, 200.0, 300.0]}))

    save_scaler(btc_scaler, ticker="BTC-USD")
    save_scaler(eth_scaler, ticker="ETH-USD")

    loaded_btc = load_scaler(ticker="BTC-USD")
    loaded_eth = load_scaler(ticker="ETH-USD")

    assert loaded_btc.mean_[0] == pytest.approx(1.0)
    assert loaded_eth.mean_[0] == pytest.approx(200.0)
    assert (tmp_path / "artifacts" / "scaler.joblib").exists()  # BTC-USD backward-compat name
    assert (tmp_path / "artifacts" / "scaler_ETH-USD.joblib").exists()


def test_load_scaler_missing_raises_file_not_found(monkeypatch, tmp_path):
    import src.data.scaling as scaling_mod

    monkeypatch.setattr(scaling_mod, "data_dir", lambda: tmp_path)
    (tmp_path / "artifacts").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_scaler(ticker="DOES-NOT-EXIST")
