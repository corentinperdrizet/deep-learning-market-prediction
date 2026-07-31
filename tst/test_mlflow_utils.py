"""Regression test for a real bug found in review: run_transformer_mlflow.py
logged the full test-report dict (which now includes nested `meta`,
`model_config`, `train_summary` dicts and a `features` list after
harmonizing the report schema with run_lstm.py) straight into
MLflowTracker.log_metrics(), which requires plain floats -> crashed with
TypeError: float() argument must be ... not 'dict'.
"""

from __future__ import annotations

from src.track.mlflow_utils import flatten_numeric_metrics


def test_flatten_numeric_metrics_flattens_nested_dicts():
    report = {
        "test_pr_auc": 0.52,
        "test_roc_auc": 0.51,
        "meta": {"ticker": "BTC-USD", "horizon": 1},
        "model_config": {"hidden_size": 128, "dropout": 0.2},
    }
    flat = flatten_numeric_metrics(report, prefix="test")
    assert flat["test/test_pr_auc"] == 0.52
    assert flat["test/test_roc_auc"] == 0.51
    assert flat["test/meta/horizon"] == 1.0
    assert flat["test/model_config/hidden_size"] == 128.0
    assert flat["test/model_config/dropout"] == 0.2


def test_flatten_numeric_metrics_drops_non_numeric_leaves():
    report = {
        "features": ["log_ret", "rsi_14"],
        "meta": {"ticker": "BTC-USD", "label_type": "direction"},
        "score": 0.5,
    }
    flat = flatten_numeric_metrics(report)
    assert flat == {"score": 0.5}


def test_flatten_numeric_metrics_drops_non_finite_and_bools():
    report = {"a": float("nan"), "b": float("inf"), "c": True, "d": 1}
    flat = flatten_numeric_metrics(report)
    assert flat == {"d": 1.0}
