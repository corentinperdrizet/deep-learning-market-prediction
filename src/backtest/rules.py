# src/backtest/rules.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_series(x, index=None, name=None) -> pd.Series:
    """
    Convert array-like to a pandas Series with an optional index and name.
    If already a Series, copy and (optionally) rename it.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        if name is not None:
            s.name = name
        return s
    x = np.asarray(x)
    return pd.Series(x, index=index, name=name)

def signal_from_proba(
    p_up,
    theta: float,
    long_short: bool = False,
    index=None,
    name: str = "signal",
) -> pd.Series:
    """
    Classification → turn probability p(up) into a desired position signal.

    Modes:
      - long_short = False (default): long if p > θ, else flat → {0, +1}
      - long_short = True: long if p > θ, short if p < 1-θ, else flat → {-1, 0, +1}

    Notes:
      - 'Desired' means pre-execution; the engine applies a +1-bar shift.
      - θ must be selected on validation and reused on test (no leakage).
    """
    p = _to_series(p_up, index=index, name="p_up")
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1).")
    if not long_short:
        sig = (p > theta).astype(float)  # {0.0, 1.0}
    else:
        sig = pd.Series(np.zeros_like(p, dtype=float), index=p.index)
        sig[p > theta] = 1.0
        sig[p < (1.0 - theta)] = -1.0
    sig.name = name
    return sig

# def signal_from_proba(p_up: pd.Series, theta: float, long_short: bool = False) -> pd.Series:
#     """
#     Convert model probabilities to trading signals.
#     long_short=False → 1=long, 0=flat
#     long_short=True  → 1=long, -1=short
#     """
#     if long_short:
#         return pd.Series(np.where(p_up > theta, 1, -1), index=p_up.index, name="signal")
#     else:
#         return pd.Series(np.where(p_up > theta, 1, 0), index=p_up.index, name="signal")


def signal_from_regression(
    y_pred,
    k: float = 1.0,
    clip: float = 1.0,
    index=None,
    name: str = "signal",
) -> pd.Series:
    """
    Regression → proportional position from predicted next return.

    position_t = clip( k * y_pred_t, [-clip, +clip] )

    Notes:
      - Use 'k' to scale aggressiveness (k=1.0 by default).
      - 'clip' bounds leverage (clip=1.0 → max |pos| = 1).
      - Feed this as 'desired' position; engine will shift +1 bar.
    """
    s = _to_series(y_pred, index=index, name="y_pred")
    pos = k * s
    if clip is not None:
        pos = pos.clip(lower=-abs(clip), upper=abs(clip))
    pos.name = name
    return pos
