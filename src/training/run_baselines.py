# src/training/run_baselines.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.config import DataConfig

# --- Data imports
from src.data.dataset import prepare_dataset
from src.data.loaders import load_prices  # pd.Series with DateTimeIndex
from src.training.evaluate import run_baselines


def _build_cfg(args):
    """
    Return a DataConfig suitable for prepare_dataset(cfg=...), built directly
    from the CLI args so it always matches the price series loaded for the
    SMA baseline (same ticker/interval/start/end/test_start).

    A --config YAML path, if provided, takes priority and is returned as-is
    (advanced/exotic setups only).
    """
    if args.config is not None:
        try:
            import yaml  # type: ignore
        except Exception:
            print(
                "[run_baselines] Warning: PyYAML not installed; ignoring --config and using CLI args.",
                file=sys.stderr,
            )
        else:
            with open(args.config) as f:
                return yaml.safe_load(f)

    return DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        end=args.end,
        test_start=args.test_start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default="2023-01-01")
    parser.add_argument("--use-xgb", action="store_true")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "flatten_last_k"])
    parser.add_argument("--out", type=Path, default=Path("data/artifacts/baselines_metrics.csv"))
    parser.add_argument("--config", type=str, default=None, help="Path to data config YAML (optional)")
    args = parser.parse_args()

    cfg = _build_cfg(args)
    dataset = prepare_dataset(cfg)

    # Load a price series aligned with the SAME (ticker, interval, start, end)
    # used to build the dataset, so SMA baseline dates match the val/test index.
    px = load_prices(args.ticker, args.interval, start=args.start, end=args.end)

    df = run_baselines(dataset, prices=px, use_xgb=args.use_xgb, pooling_lr=args.pooling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nBaseline metrics saved to:", args.out)
    print(df)


if __name__ == "__main__":
    main()
