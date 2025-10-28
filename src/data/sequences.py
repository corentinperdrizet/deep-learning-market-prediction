# -----------------------------
# File: src/data/sequences.py
# -----------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Create (N, seq_len, F) sequences and aligned labels without look-ahead.
    y at time t corresponds to sequence [t-seq_len ... t-1].
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


