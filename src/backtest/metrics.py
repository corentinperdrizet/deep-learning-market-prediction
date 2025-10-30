# src/backtest/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict

def _periods_per_year(index: pd.Index, fallback: int = 365) -> int:
    """
    Infer periods per year from the index (rough heuristic):
      - If it looks like business days → ~252
      - Else assume 365 (crypto daily)
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) > 3:
        days = (index[-1] - index[0]).days
        if days > 0:
            unique_days = len(pd.DatetimeIndex(index.date).unique())
            approx_years = days / 365.25 if days else 1.0
            if approx_years > 0:
                per_year = unique_days / approx_years
                if 225 <= per_year <= 280:
                    return 252
    return fallback

def cagr(returns: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """
    CAGR = (prod(1 + r))^(periods_per_year / N) - 1
    """
    r = returns.dropna().astype(float)
    if periods_per_year is None:
        periods_per_year = _periods_per_year(r.index)
    total_return = (1.0 + r).prod()
    n = r.shape[0]
    if n == 0:
        return np.nan
    return total_return ** (periods_per_year / n) - 1.0

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: Optional[int] = None) -> float:
    """
    Annualized Sharpe ratio using sample std dev:
      Sharpe = sqrt(PY) * (mean(r - rf/PY) / std(r - rf/PY))
    """
    r = returns.dropna().astype(float)
    if periods_per_year is None:
        periods_per_year = _periods_per_year(r.index)
    ex = r - rf / periods_per_year
    mu, sigma = ex.mean(), ex.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return np.sqrt(periods_per_year) * (mu / sigma)

def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: Optional[int] = None) -> float:
    """
    Annualized Sortino ratio with downside deviation.
    """
    r = returns.dropna().astype(float)
    if periods_per_year is None:
        periods_per_year = _periods_per_year(r.index)
    ex = r - rf / periods_per_year
    downside = ex.copy()
    downside[downside > 0] = 0.0
    dd_sigma = downside.std(ddof=1)
    if dd_sigma == 0 or np.isnan(dd_sigma):
        return np.nan
    return np.sqrt(periods_per_year) * (ex.mean() / dd_sigma)

def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown: min over time of (equity / rolling_max - 1)
    """
    e = equity.dropna().astype(float)
    run_max = e.cummax()
    dd = e / run_max - 1.0
    return dd.min()

def calmar(returns: pd.Series, equity: Optional[pd.Series] = None, periods_per_year: Optional[int] = None) -> float:
    """
    Calmar ratio = CAGR / |Max Drawdown|
    """
    c = cagr(returns, periods_per_year=periods_per_year)
    if equity is None:
        equity = (1.0 + returns.dropna()).cumprod()
    mdd = abs(max_drawdown(equity))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return c / mdd

def turnover(turn: pd.Series) -> float:
    """
    Average turnover (mean absolute position change per period).
    """
    return float(turn.dropna().mean())

def hit_ratio(returns: pd.Series) -> float:
    """
    Fraction of periods with positive net return.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan
    return float((r > 0).mean())

def summary_kpis(
    bt_df: pd.DataFrame,
    ret_col: str = "ret_net",
    turnover_col: str = "turnover",
    equity_col: str = "equity_net",
    periods_per_year: Optional[int] = None,
    rf: float = 0.0,
) -> Dict[str, float]:
    """
    Compute a dictionary of key portfolio metrics for a backtest dataframe.
    """
    r = bt_df[ret_col].dropna()
    if periods_per_year is None:
        periods_per_year = _periods_per_year(r.index)
    eq = bt_df[equity_col]
    kpis = {
        "CAGR": cagr(r, periods_per_year),
        "Sharpe": sharpe(r, rf=rf, periods_per_year=periods_per_year),
        "Sortino": sortino(r, rf=rf, periods_per_year=periods_per_year),
        "MaxDrawdown": max_drawdown(eq),
        "Calmar": calmar(r, equity=eq, periods_per_year=periods_per_year),
        "Turnover": turnover(bt_df[turnover_col]),
        "HitRatio": hit_ratio(r),
    }
    return kpis
