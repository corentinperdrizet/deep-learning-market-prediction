# =============================
# Project: IA - Tx — Data Layer (src/data)
# Scope: End-to-end data utilities for BTC-USD (daily), modular for future assets/frequencies.
# Files included in this canvas (copy them to your repo under src/data/):
#   - __init__.py
#   - paths.py
#   - config.py
#   - loaders.py
#   - quality.py
#   - features.py
#   - preprocessing.py
#   - scaling.py
#   - sequences.py
#   - viz.py
#   - dataset.py  (orchestrator)
#
# External deps: pandas, numpy, yfinance, scikit-learn, matplotlib, joblib
# Optional: plotnine or seaborn (not required)
# =============================

# -----------------------------
# File: src/data/__init__.py
# -----------------------------

"""Data layer package for market prediction project.

This package provides:
- Robust loading of market data (currently BTC-USD daily via yfinance)
- Data quality checks (missingness, duplicate timestamps, calendar gaps)
- Feature engineering (returns, volatility, RSI, MACD, rolling stats, calendar)
- Labeling (directional next-step or h-step ahead returns)
- Safe scaling (fit on train only) with persisted scalers
- Sequence construction for DL models (LSTM/GRU/Transformer)
- Visualization helpers for quick EDA
- A high-level orchestrator to build processed datasets

Design goals:
- Deterministic, reproducible, testable
- No look-ahead: all rolling windows are strictly causal
- Config-driven to ease future extension to ETH, S&P500, other frequencies
"""

__all__ = [
    "paths", "config", "loaders", "quality", "features",
    "preprocessing", "scaling", "sequences", "viz", "dataset"
]


