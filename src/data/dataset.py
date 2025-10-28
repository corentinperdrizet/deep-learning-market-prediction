# -----------------------------
# File: src/data/dataset.py
# -----------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import DataConfig
from .loaders import download_ohlcv, ensure_daily_calendar
from .quality import check_missingness, check_duplicates, detect_calendar_gaps
from .preprocessing import build_features, make_label, FEATURE_COLUMNS_DEFAULT, time_splits
from .scaling import fit_scaler, transform_with_scaler, save_scaler
from .sequences import build_sequences
from .paths import data_dir


def prepare_dataset(cfg: DataConfig, seq_len: int = 64) -> Dict[str, Any]:
    """End-to-end preparation of dataset ready for modeling.

    Returns a dict with:
      - X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
      - features: list of feature names
      - idx: dict of index ranges for splits
      - meta: info about ticker/interval/labeling
    Saves scaler artifact and processed parquet files if configured.
    """
    # 1) Load
    df = download_ohlcv(cfg.ticker, cfg.start, cfg.end, cfg.interval, cache=cfg.cache_raw)
    df = ensure_daily_calendar(df)

    # 2) Quality checks (print summary)
    miss = check_missingness(df)
    dups = check_duplicates(df)
    gaps = detect_calendar_gaps(df)

    print("[Quality] Missing per col:\n", miss)
    print("[Quality] Duplicate timestamps:", dups)
    if not gaps.empty:
        print(f"[Quality] Detected {len(gaps)} calendar gaps (freq=1D)")

    # 3) Features
    feats = build_features(
        df,
        ret_windows=cfg.ret_windows,
        vol_window=cfg.vol_window,
        rsi_window=cfg.rsi_window,
        macd_fast=cfg.macd_fast,
        macd_slow=cfg.macd_slow,
        macd_signal=cfg.macd_signal,
    )

    # 4) Label
    y = make_label(df, label_type=cfg.label_type, horizon=cfg.horizon)

    # 5) Align & drop NA
    data = pd.concat([df[["Close"]], feats, y.rename("target")], axis=1)
    data = data.dropna()

    # 6) Train/Val/Test split by time
    tr_idx, va_idx, te_idx = time_splits(data, cfg.val_start, cfg.test_start)
    features = [c for c in FEATURE_COLUMNS_DEFAULT if c in data.columns]

    X_train_df = data.loc[tr_idx, features]
    X_val_df = data.loc[va_idx, features]
    X_test_df = data.loc[te_idx, features]
    y_train = data.loc[tr_idx, "target"].values
    y_val = data.loc[va_idx, "target"].values
    y_test = data.loc[te_idx, "target"].values

    # 7) Scaling (fit on train only)
    scaler = fit_scaler(X_train_df, robust=cfg.use_robust_scaler)
    save_scaler(scaler)
    X_train = transform_with_scaler(scaler, X_train_df)
    X_val = transform_with_scaler(scaler, X_val_df)
    X_test = transform_with_scaler(scaler, X_test_df)

    # 8) Sequences
    X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len=seq_len)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, seq_len=seq_len)

    # 9) Save processed (optional)
    if cfg.cache_processed:
        outdir = data_dir() / "processed"
        pd.DataFrame(data).to_parquet(outdir / f"{cfg.ticker.replace('/', '-')}_{cfg.interval}_dataset.parquet")

    # 10) Pack results
    result = {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_val": X_val_seq,
        "y_val": y_val_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "features": features,
        "idx": {"train": tr_idx, "val": va_idx, "test": te_idx},
        "meta": {
            "ticker": cfg.ticker,
            "interval": cfg.interval,
            "label_type": cfg.label_type,
            "horizon": cfg.horizon,
            "seq_len": seq_len,
        },
    }
    return result


if __name__ == "__main__":
    # Quick smoke test when running directly
    cfg = DataConfig(
        ticker="BTC-USD",
        interval="1d",
        start="2018-01-01",
        end=None,
        label_type="direction",
        horizon=1,
        test_start="2023-01-01",
    )
    out = prepare_dataset(cfg, seq_len=64)
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, type(v))
