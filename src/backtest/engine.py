# src/backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .costs import per_period_cost_from_turnover

@dataclass
class BacktestResult:
    """
    Container for the full backtest output.
    'df' includes:
      - ret_asset: simple returns of the asset
      - signal: desired (pre-execution) signal
      - pos: executed position (signal shifted +1 bar, clipped)
      - turnover: absolute position change
      - cost: transaction cost per period
      - ret_gross: gross strategy return (pos * ret_asset)
      - ret_net: net strategy return after costs
      - equity_net: cumulative equity (product of 1+ret_net)
      - drawdown: equity / rolling max - 1
    """
    df: pd.DataFrame
    name: str = "strategy"

def _to_series(x, index=None, name=None) -> pd.Series:
    """
    Convert array-like to a pandas Series with an optional index and name.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        if name is not None:
            s.name = name
        return s
    x = np.asarray(x)
    return pd.Series(x, index=index, name=name)

def backtest(
    ret_asset,              # pandas Series of simple returns (preferred)
    signal_desired,         # desired position Series before execution shift
    fees_bps: float = 10.0,
    slippage_bps: float = 0.0,
    max_abs_pos: float = 1.0,
    name: str = "strategy",
) -> BacktestResult:
    """
    Execute a simple long-only / long-short strategy with +1 bar delay.

    Execution model:
      pos_t = clip(signal_{t-1}, [-max_abs_pos, +max_abs_pos])

    PnL accounting:
      ret_gross_t = pos_t * ret_asset_t
      turnover_t  = |pos_t - pos_{t-1}|
      cost_t      = turnover_t * (fees_bps + slippage_bps)/1e4
      ret_net_t   = ret_gross_t - cost_t

    Risk stats helpers:
      equity_net_t = cumprod(1 + ret_net_t)
      drawdown_t   = equity_net_t / cummax(equity_net_t) - 1
    """
    r = _to_series(ret_asset, name="ret_asset").astype(float)
    s = _to_series(signal_desired, index=r.index, name="signal_desired").astype(float)

    if not s.index.equals(r.index):
        s = s.reindex(r.index)
    
    # +1 bar shift to avoid look-ahead bias
    pos = s.shift(1).fillna(0.0)
    if max_abs_pos is not None:
        pos = pos.clip(lower=-abs(max_abs_pos), upper=abs(max_abs_pos))
    pos.name = "pos"

    # Turnover and costs
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    turnover.name = "turnover"

    cost = per_period_cost_from_turnover(
        turnover, fees_bps=fees_bps, slippage_bps=slippage_bps, name="cost"
    )

    # PnL
    ret_gross = (pos * r).rename("ret_gross")
    ret_net = (ret_gross - cost).rename("ret_net")

    # Equity and drawdown
    equity_net = (1.0 + ret_net).cumprod().rename("equity_net")
    running_max = equity_net.cummax()
    drawdown = (equity_net / running_max - 1.0).rename("drawdown")

    df = pd.concat(
        [
            r,
            s.rename("signal"),
            pos,
            turnover,
            cost,
            ret_gross,
            ret_net,
            equity_net,
            drawdown,
        ],
        axis=1,
    )

    return BacktestResult(df=df, name=name)
