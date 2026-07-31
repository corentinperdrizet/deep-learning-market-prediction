from __future__ import annotations

import pandas as pd

from src.training.run_multi_asset import _artifact_prefix, plot_multi_asset_comparison, run_multi_asset


def test_artifact_prefix_keeps_btc_backward_compatible():
    assert _artifact_prefix("BTC-USD") == "lstm"


def test_artifact_prefix_is_ticker_specific_for_others():
    assert _artifact_prefix("ETH-USD") == "lstm_ETH-USD"
    assert _artifact_prefix("^GSPC") == "lstm_GSPC"


def test_run_multi_asset_records_failures_without_crashing_the_sweep(monkeypatch, tmp_path):
    """If one ticker's training blows up (bad data, network hiccup, whatever),
    the sweep must keep going and record the failure rather than crash.
    """
    import src.training.run_multi_asset as rma_mod

    def fake_train_lstm(cfg, **kwargs):
        if cfg.ticker == "BAD-TICKER":
            raise RuntimeError("simulated failure")
        return {
            "meta": {"ticker": cfg.ticker},
            "train_summary": {"best_epoch": 3},
            "test_roc_auc": 0.55,
            "test_pr_auc": 0.52,
        }

    monkeypatch.setattr(rma_mod, "train_lstm", fake_train_lstm)

    df = run_multi_asset(tickers=["BTC-USD", "BAD-TICKER"], artifacts_dir=str(tmp_path))

    assert len(df) == 2
    ok_row = df[df["ticker"] == "BTC-USD"].iloc[0]
    bad_row = df[df["ticker"] == "BAD-TICKER"].iloc[0]
    assert ok_row["error"] is None
    assert ok_row["test_roc_auc"] == 0.55
    assert bad_row["error"] == "simulated failure"
    assert pd.isna(bad_row["test_roc_auc"])


def test_plot_multi_asset_comparison_skips_failed_rows(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["BTC-USD", "BAD-TICKER"],
            "test_roc_auc": [0.55, None],
            "test_pr_auc": [0.52, None],
        }
    )
    outpath = tmp_path / "comparison.png"
    plot_multi_asset_comparison(df, str(outpath))
    assert outpath.exists()
