from __future__ import annotations

import numpy as np

from src.data.sequences import build_sequences


def test_build_sequences_shapes():
    n, f, seq_len = 50, 4, 10
    X = np.arange(n * f, dtype=float).reshape(n, f)
    y = np.arange(n, dtype=float)

    Xs, ys = build_sequences(X, y, seq_len=seq_len)

    assert Xs.shape == (n - seq_len, seq_len, f)
    assert ys.shape == (n - seq_len,)


def test_build_sequences_excludes_current_row_no_leakage():
    """y[i] must be paired with X[i-seq_len:i], i.e. the window strictly
    precedes the label's own row -- the model must never see the features
    of the day it's predicting.
    """
    n, seq_len = 30, 5
    X = np.arange(n).reshape(n, 1).astype(float)  # row index as the only feature
    y = np.arange(n).astype(float)

    Xs, ys = build_sequences(X, y, seq_len=seq_len)

    for k in range(len(ys)):
        target_row = seq_len + k
        window_last_row = Xs[k, -1, 0]
        assert window_last_row == target_row - 1
        assert ys[k] == target_row


def test_build_sequences_empty_when_shorter_than_seq_len():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    Xs, ys = build_sequences(X, y, seq_len=10)
    assert Xs.shape == (0,)
    assert ys.shape == (0,)
