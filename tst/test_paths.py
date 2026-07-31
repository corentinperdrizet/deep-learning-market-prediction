from __future__ import annotations

from src.data.paths import artifact_prefix, scaler_filename


def test_scaler_filename_backward_compatible_for_btc():
    assert scaler_filename("BTC-USD") == "scaler.joblib"


def test_scaler_filename_ticker_suffixed_for_other_assets():
    assert scaler_filename("ETH-USD") == "scaler_ETH-USD.joblib"
    assert scaler_filename("^GSPC") == "scaler_GSPC.joblib"


def test_artifact_prefix_backward_compatible_for_btc():
    assert artifact_prefix("lstm", "BTC-USD") == "lstm"
    assert artifact_prefix("transformer", "BTC-USD") == "transformer"


def test_artifact_prefix_ticker_suffixed_for_other_assets():
    assert artifact_prefix("lstm", "ETH-USD") == "lstm_ETH-USD"
    assert artifact_prefix("transformer", "^GSPC") == "transformer_GSPC"
