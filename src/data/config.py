# -----------------------------
# File: src/data/config.py
# -----------------------------
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


TaskType = Literal["classification", "regression"]


@dataclass
class DataConfig:
    # Asset + timeframe
    ticker: str = "BTC-USD"
    interval: Literal["1d"] = "1d"  # ready for future extension (e.g., "1h")
    start: str = "2018-01-01"
    end: Optional[str] = None  # None = up to latest

    # Labeling
    label_type: Literal["direction", "return"] = "direction"
    horizon: int = 1  # steps ahead (1 day)

    # Feature windows
    vol_window: int = 20
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ret_windows: Sequence[int] = field(default_factory=lambda: [1, 3, 7, 14])

    # Split
    val_start: Optional[str] = None  # if None, will compute proportionally
    test_start: Optional[str] = "2023-01-01"

    # Paths
    cache_raw: bool = True
    cache_processed: bool = True

    # Scaling
    use_robust_scaler: bool = False  # if True use RobustScaler else StandardScaler

    # Repro
    seed: int = 42


