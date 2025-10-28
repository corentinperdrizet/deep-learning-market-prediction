# src/training/run_baselines.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import pandas as pd

# --- Data imports
from src.data.dataset import prepare_dataset

# Optional helpers that may or may not exist in your repo
try:
    from src.data.loaders import load_prices  # pd.Series with DateTimeIndex
except Exception:
    load_prices = None

# Fallback bridge to construct a default cfg if your pipeline requires one
try:
    from src.data.config_bridge import make_default_cfg  # added below
except Exception:
    make_default_cfg = None

from src.training.evaluate import run_baselines


def _build_cfg(args):
    """Return a config object/dict suitable for prepare_dataset(cfg=...).
    Priority:
    1) --config path (YAML) if provided
    2) src.data.config_bridge.make_default_cfg if available
    3) a minimal dict with common keys
    """
    # 1) Try YAML path
    if args.config is not None:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # 2) Try bridge (handles dataclass/DataConfig etc.)
    if make_default_cfg is not None:
        try:
            return make_default_cfg(ticker=args.ticker, interval=args.interval)
        except Exception:
            pass

    # 3) Minimal dict fallback (align keys with your pipeline)
    return dict(
        ticker=args.ticker,
        interval=args.interval,
        label_type="direction",
        horizon=1,
        seq_len=64,
        scaler="standard",
        # If your dataset expects explicit dates, set None and let it infer
        val_start=None,
        test_start=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--use-xgb", action="store_true")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "flatten_last_k"])
    parser.add_argument("--out", type=Path, default=Path("data/artifacts/baselines_metrics.csv"))
    parser.add_argument("--config", type=str, default=None, help="Path to data config YAML (optional)")
    args = parser.parse_args()

    # Build cfg and call prepare_dataset(cfg=...)
    cfg = _build_cfg(args)
    try:
        dataset = prepare_dataset(cfg)
    except TypeError:
        # Some pipelines accept both signatures; retry without cfg
        dataset = prepare_dataset()

    # Try to load a price series aligned with your dataset index
    px = None
    if load_prices is not None:
        try:
            px = load_prices(args.ticker, args.interval)
        except Exception:
            px = None

    # Some pipelines embed prices in the dataset dict; attempt to use them
    if px is None:
        for key in ("prices", "px", "close", "close_price"):
            if key in dataset:
                try:
                    px = dataset[key]
                    break
                except Exception:
                    pass

    df = run_baselines(dataset, prices=px, use_xgb=args.use_xgb, pooling_lr=args.pooling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Baseline metrics saved to:", args.out)
    print(df)


if __name__ == "__main__":
    main()