# src/data/config_bridge.py
"""Tiny utility to construct a default cfg compatible with your data pipeline.

It tries, in order:
- get_default_config() function
- DEFAULT_CONFIG constant
- DataConfig dataclass
- load_config() function
- finally returns a plain dict with sensible defaults
"""
from __future__ import annotations
from typing import Any


def make_default_cfg(ticker: str = "BTC-USD", interval: str = "1d") -> Any:
    # Try to import your config module
    try:
        from . import config as cfgmod  # type: ignore
    except Exception:
        cfgmod = None

    # 1) get_default_config()
    if cfgmod is not None and hasattr(cfgmod, "get_default_config"):
        return cfgmod.get_default_config()

    # 2) DEFAULT_CONFIG
    if cfgmod is not None and hasattr(cfgmod, "DEFAULT_CONFIG"):
        return getattr(cfgmod, "DEFAULT_CONFIG")

    # 3) DataConfig dataclass
    if cfgmod is not None and hasattr(cfgmod, "DataConfig"):
        try:
            return cfgmod.DataConfig(ticker=ticker, interval=interval)
        except Exception:
            # Fallback to no-arg construction
            try:
                return cfgmod.DataConfig()
            except Exception:
                pass

    # 4) load_config()
    if cfgmod is not None and hasattr(cfgmod, "load_config"):
        try:
            return cfgmod.load_config()
        except Exception:
            pass

    # 5) plain dict fallback
    return {
        "ticker": ticker,
        "interval": interval,
        "label_type": "direction",
        "horizon": 1,
        "seq_len": 64,
        "scaler": "standard",
        "val_start": None,
        "test_start": None,
    }