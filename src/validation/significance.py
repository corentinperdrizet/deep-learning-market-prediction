# -----------------------------
# File: src/validation/significance.py
# -----------------------------
"""Statistical rigor on top of a trained model's backtest: is the Sharpe
ratio distinguishable from noise, and how sensitive is it to trading costs?

A single point-estimate Sharpe on a few hundred test-set days is easy to
over-interpret. This module adds three standard quant-research tools:
  - a block bootstrap confidence interval on the Sharpe ratio (block, not
    i.i.d., because daily returns are autocorrelated -- an i.i.d. bootstrap
    would understate the true uncertainty),
  - the Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012), which
    gives P(true Sharpe > benchmark) adjusted for sample size, skew and
    kurtosis instead of just eyeballing a point estimate,
  - a transaction-cost sensitivity sweep, since the whole edge can vanish
    once realistic fees are applied.

run_significance_for_lstm() also re-optimizes and persists the LSTM's
decision threshold theta on the validation split before evaluating on test
(fixing a real issue found in review: the checkpointed-in thresholds.json
held a stale theta=0.05 from a much older run of the pipeline, silently out
of sync with the current model/data). As a byproduct it writes
data/artifacts/lstm_signals.csv, which src/app/streamlit_app.py's "Signals"
tab already knows how to read but nothing previously produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..backtest.engine import backtest
from ..backtest.metrics import _periods_per_year, sharpe, summary_kpis
from ..backtest.plots import plot_drawdown, plot_equity
from ..backtest.rules import signal_from_proba
from ..data.config import DataConfig
from ..data.dataset import build_feature_frame, prepare_dataset
from ..models.lstm import LSTMClassifier
from ..training.thresholds import grid_search_threshold
from ..training.utils import ensure_dir, get_device


def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    block_size: int = 10,
    ci: float = 0.95,
    periods_per_year: int | None = None,
    seed: int = 0,
) -> dict:
    """Circular block bootstrap confidence interval on the annualized Sharpe ratio."""
    r = returns.dropna().astype(float)
    n = len(r)
    if periods_per_year is None:
        periods_per_year = _periods_per_year(r.index)
    observed = float(sharpe(r, periods_per_year=periods_per_year))

    values = r.to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    boot_sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([np.take(values, range(s, s + block_size), mode="wrap") for s in starts])[:n]
        mu, sigma = sample.mean(), sample.std(ddof=1)
        if sigma < 1e-12:
            continue
        boot_sharpes.append(np.sqrt(periods_per_year) * mu / sigma)

    boot_sharpes = np.asarray(boot_sharpes)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return {
        "observed_sharpe": observed,
        "ci_low": float(np.percentile(boot_sharpes, lo_pct)) if len(boot_sharpes) else None,
        "ci_high": float(np.percentile(boot_sharpes, hi_pct)) if len(boot_sharpes) else None,
        "n_boot_valid": int(len(boot_sharpes)),
        "block_size": block_size,
        "confidence": ci,
    }


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey & Lopez de Prado's Probabilistic Sharpe Ratio: P(true Sharpe > benchmark_sharpe).

    `kurtosis` uses the normal-distribution convention (kurtosis=3 for a
    Gaussian), NOT pandas' excess-kurtosis convention (which is 0 for a
    Gaussian) -- callers passing pandas .kurtosis() must add 3.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    numerator = (observed_sharpe - benchmark_sharpe) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2)
    if denominator <= 0 or np.isnan(denominator):
        return float("nan")
    return float(stats.norm.cdf(numerator / denominator))


def cost_sensitivity(
    ret_asset: pd.Series,
    signal: pd.Series,
    fee_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0),
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Sweep transaction costs and report how the strategy's KPIs degrade."""
    rows = []
    for fee in fee_bps_grid:
        res = backtest(ret_asset, signal, fees_bps=fee, slippage_bps=slippage_bps)
        kpis = summary_kpis(res.df)
        rows.append({"fees_bps": fee, **kpis})
    return pd.DataFrame(rows)


def _json_safe(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def run_significance_for_lstm(
    cfg: DataConfig,
    seq_len: int = 64,
    fees_bps: float = 10.0,
    ckpt_path: str = "data/artifacts/lstm_classifier.pt",
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    n_boot: int = 1000,
    block_size: int = 10,
) -> dict:
    data = prepare_dataset(cfg, seq_len=seq_len)
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    idx_val = data["idx"]["val"][seq_len:]
    idx_test = data["idx"]["test"][seq_len:]

    device = get_device()
    model = LSTMClassifier(
        input_dim=X_val.shape[2],
        hidden_size=hidden,
        num_layers=layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits_val = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        logits_test = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
    p_val = 1.0 / (1.0 + np.exp(-logits_val))
    p_test = 1.0 / (1.0 + np.exp(-logits_test))

    close = build_feature_frame(cfg)["Close"]
    ret_val = close.pct_change().reindex(idx_val).to_numpy()
    ret_test = close.pct_change().reindex(idx_test)

    # Re-optimize theta on validation (Sharpe objective), then freeze it for test.
    theta_result = grid_search_threshold(y_val, p_val, objective="sharpe", returns_next=ret_val)
    theta = theta_result.best_threshold

    thresholds_path = Path("data/artifacts/thresholds.json")
    payload = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}
    payload["lstm"] = {
        "theta": theta,
        "objective": "sharpe",
        "val_criterion_value": theta_result.criterion_value,
    }
    ensure_dir(str(thresholds_path.parent))
    thresholds_path.write_text(json.dumps(payload, indent=2))

    signal_test = signal_from_proba(pd.Series(p_test, index=idx_test), theta=theta)
    bt = backtest(ret_test, signal_test, fees_bps=fees_bps)
    kpis = summary_kpis(bt.df)

    buy_hold = backtest(ret_test, pd.Series(1.0, index=idx_test), fees_bps=0.0)
    bt.df["equity_bh"] = buy_hold.df["equity_net"]

    artifact_prefix = "lstm"  # this driver is LSTM-only for now (see module docstring)
    kpis_path = Path("data/artifacts") / f"{artifact_prefix}_backtest_kpis.csv"
    ensure_dir(str(kpis_path.parent))
    pd.DataFrame([kpis]).to_csv(kpis_path, index=False)

    plot_equity(bt.df, equity_bh_col="equity_bh", title=f"{cfg.ticker} LSTM strategy vs Buy & Hold")
    equity_fig_path = Path("experiments/figures") / f"{artifact_prefix}_equity.png"
    ensure_dir(str(equity_fig_path.parent))
    plt.savefig(equity_fig_path, dpi=150)
    plt.close()

    plot_drawdown(bt.df, title=f"{cfg.ticker} LSTM strategy drawdown")
    dd_fig_path = Path("experiments/figures") / f"{artifact_prefix}_drawdown.png"
    plt.savefig(dd_fig_path, dpi=150)
    plt.close()

    net_returns = bt.df["ret_net"].dropna()
    boot = block_bootstrap_sharpe_ci(net_returns, n_boot=n_boot, block_size=block_size)
    skew = float(net_returns.skew())
    kurt_normal_convention = float(net_returns.kurtosis()) + 3.0  # pandas uses excess kurtosis
    psr = (
        probabilistic_sharpe_ratio(
            kpis["Sharpe"],
            benchmark_sharpe=0.0,
            n=len(net_returns),
            skew=skew,
            kurtosis=kurt_normal_convention,
        )
        if not np.isnan(kpis["Sharpe"])
        else None
    )

    cost_df = cost_sensitivity(ret_test, signal_test)
    cost_path = Path("data/artifacts/lstm_cost_sensitivity.csv")
    cost_df.to_csv(cost_path, index=False)

    signals_df = pd.DataFrame(
        {
            "timestamp": idx_test,
            "price": close.reindex(idx_test).to_numpy(),
            "p_up": p_test,
            "y_true": y_test,
        }
    )
    signals_path = Path("data/artifacts/lstm_signals.csv")
    signals_df.to_csv(signals_path, index=False)

    report = {
        "meta": {
            "ticker": cfg.ticker,
            "theta": theta,
            "theta_objective": "sharpe (re-optimized on validation, frozen for test)",
            "fees_bps": fees_bps,
            "n_test_periods": int(len(net_returns)),
        },
        "test_kpis": kpis,
        "bootstrap_sharpe_ci": boot,
        "probabilistic_sharpe_ratio_vs_zero": psr,
        "cost_sensitivity": cost_df.to_dict(orient="records"),
        "artifacts": {
            "signals_csv": str(signals_path),
            "cost_sensitivity_csv": str(cost_path),
            "kpis_csv": str(kpis_path),
            "equity_fig": str(equity_fig_path),
            "drawdown_fig": str(dd_fig_path),
        },
    }
    return _json_safe(report)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Statistical significance + cost sensitivity for the LSTM")
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--ckpt", type=str, default="data/artifacts/lstm_classifier.pt")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=10)
    p.add_argument("--out", type=str, default="data/artifacts/significance_report.json")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        test_start=args.test_start,
        horizon=args.horizon,
    )
    report = run_significance_for_lstm(
        cfg,
        seq_len=args.seq_len,
        fees_bps=args.fees_bps,
        ckpt_path=args.ckpt,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        n_boot=args.n_boot,
        block_size=args.block_size,
    )
    ensure_dir(str(Path(args.out).parent))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved significance report -> {args.out}")
    print(json.dumps({k: v for k, v in report.items() if k != "cost_sensitivity"}, indent=2))


if __name__ == "__main__":
    main()
