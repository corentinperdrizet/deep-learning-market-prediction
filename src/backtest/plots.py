# src/backtest/plots.py
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity(
    df: pd.DataFrame,
    equity_col: str = "equity_net",
    equity_bh_col: str | None = None,
    title: str = "Equity Curve",
):
    """
    Plot strategy equity and (optionally) Buy & Hold equity.
    """
    plt.figure()
    df[equity_col].plot(label="Strategy")
    if equity_bh_col and equity_bh_col in df.columns:
        df[equity_bh_col].plot(label="Buy & Hold")
    plt.title(title)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity (×)")
    plt.tight_layout()

def plot_drawdown(
    df: pd.DataFrame,
    drawdown_col: str = "drawdown",
    title: str = "Drawdown",
):
    """
    Plot strategy drawdown series.
    """
    plt.figure()
    df[drawdown_col].plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
