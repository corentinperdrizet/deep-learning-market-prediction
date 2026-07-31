from __future__ import annotations

import numpy as np
import pytest

from src.training.thresholds import grid_search_threshold


def test_grid_search_threshold_f1_finds_near_perfect_separator():
    """y is perfectly separated by p at 0.5 (all y=1 have p>0.5, all y=0 have
    p<0.5) -- the F1-optimal threshold must land in that gap, and F1 there
    should be ~1.0.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    p = np.concatenate([rng.uniform(0.0, 0.4, n // 2), rng.uniform(0.6, 1.0, n // 2)])
    result = grid_search_threshold(y, p, objective="f1")
    assert result.best_threshold == pytest.approx(0.4, abs=0.15)
    assert result.criterion_value > 0.95


def test_grid_search_threshold_sharpe_requires_returns():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    with pytest.raises(AssertionError):
        grid_search_threshold(y, p, objective="sharpe")


def test_grid_search_threshold_table_has_expected_columns():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4])
    result = grid_search_threshold(y, p, objective="f1")
    expected_cols = {"threshold", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "sharpe"}
    assert expected_cols.issubset(set(result.table.columns))


def test_grid_search_threshold_rejects_unknown_objective():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        grid_search_threshold(y, p, objective="not_an_objective")
