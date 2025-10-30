# src/backtest/costs.py
from __future__ import annotations
import numpy as np
import pandas as pd

def per_period_cost_from_turnover(
    turnover: pd.Series | np.ndarray,
    fees_bps: float = 10.0,
    slippage_bps: float = 0.0,
    name: str = "cost",
) -> pd.Series:
    """
    Transaction cost per period based on turnover.

    Definitions:
      - turnover_t = |pos_t - pos_{t-1}|
      - fees/slippage expressed in basis points (bps).
      - cost_t = turnover_t * (fees_bps + slippage_bps) / 1e4

    Returns:
      pandas Series of per-period costs (in return points)
      aligned with 'turnover'.
    """
    if isinstance(turnover, pd.Series):
        idx = turnover.index
        turn = turnover.values.astype(float)
    else:
        turn = np.asarray(turnover, dtype=float)
        idx = None
    bps = (fees_bps + slippage_bps) / 1e4
    cost = turn * bps
    return pd.Series(cost, index=idx, name=name)
