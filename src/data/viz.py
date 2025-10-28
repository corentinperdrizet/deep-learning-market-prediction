# -----------------------------
# File: src/data/viz.py
# -----------------------------
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def plot_price(df: pd.DataFrame, title: str = "Close"):
    df["Close"].plot(title=title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_missing(df: pd.DataFrame):
    (df.isna().sum()).plot(kind="bar", title="Missing values per column")
    plt.tight_layout()
    plt.show()


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series | None = None, title: str = "Equity Curve"):
    ax = equity.plot(label="Strategy")
    if benchmark is not None:
        benchmark.plot(ax=ax, label="Benchmark")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


