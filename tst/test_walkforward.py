from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.walkforward import WalkForwardSplitter


def _synthetic_frame(n=1000):
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"x": np.arange(n), "Close": 100.0 + np.arange(n), "target": np.arange(n) % 2}, index=idx
    )


def test_folds_are_chronological_and_expanding():
    data = _synthetic_frame()
    splitter = WalkForwardSplitter(n_splits=4, min_train_size=200, test_size=100, embargo=1)
    folds = splitter.split(data)

    assert len(folds) == 4
    for i in range(1, len(folds)):
        # Expanding window: each fold's train set is a strict superset (by end date) of the previous.
        assert folds[i].train_idx.max() > folds[i - 1].train_idx.max()
        # Folds move forward in time and don't repeat the same test window.
        assert folds[i].test_idx.min() > folds[i - 1].test_idx.min()


def test_embargo_gap_is_respected_between_train_and_test():
    data = _synthetic_frame()
    embargo = 5
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=200, test_size=90, embargo=embargo)
    folds = splitter.split(data)

    for fold in folds:
        train_end_pos = data.index.get_loc(fold.train_idx.max())
        test_start_pos = data.index.get_loc(fold.test_idx.min())
        assert test_start_pos - train_end_pos > embargo


def test_no_overlap_between_train_and_test_within_a_fold():
    data = _synthetic_frame()
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=200, test_size=90, embargo=1)
    for fold in splitter.split(data):
        assert len(fold.train_idx.intersection(fold.test_idx)) == 0


def test_too_few_rows_yields_no_folds():
    data = _synthetic_frame(n=50)
    splitter = WalkForwardSplitter(n_splits=5, min_train_size=500, test_size=90, embargo=1)
    assert splitter.split(data) == []


def test_run_walkforward_raises_clear_error_when_no_fold_fits(monkeypatch):
    """run_walkforward() must fail loudly (not silently return an empty/
    misleading report) when the requested fold sizes don't fit the data.
    build_feature_frame is monkeypatched so this stays network-free.
    """
    import src.validation.walkforward as wf_mod
    from src.data.config import DataConfig

    monkeypatch.setattr(wf_mod, "build_feature_frame", lambda cfg: _synthetic_frame(n=50))

    cfg = DataConfig(ticker="BTC-USD")
    with pytest.raises(ValueError):
        wf_mod.run_walkforward(cfg, min_train_size=500, n_splits=5, test_size=90)
