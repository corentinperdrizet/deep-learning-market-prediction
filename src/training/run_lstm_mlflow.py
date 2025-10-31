import argparse
import os
import subprocess
import sys
from typing import Dict, Any, List, Optional

from src.data.dataset import prepare_dataset
from src.data.config import DataConfig
from src.training.utils import select_device, set_seed
from src.track.mlflow_utils import MLflowTracker

ART_DIR = "data/artifacts"
FIG_DIR = "experiments/figures"

def parse_args():
    p = argparse.ArgumentParser(description="LSTM run with MLflow tracking (subprocess mode)")

    # ----------------- Data -----------------
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)

    # ----------------- Model / Train -----------------
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-weight", type=float, default=None)

    # ----------------- Misc -----------------
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--experiment", default="deep-learning-market-prediction")
    p.add_argument("--run-name", default=None)
    return p.parse_args()

import math

def _flatten_numeric(prefix, obj, out):
    """
    Recursively flatten dicts and keep only finite numeric leaf values.
    Keys are joined with '/' and prefixed with 'prefix'.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten_numeric(key, v, out)
    elif isinstance(obj, (list, tuple)):
        # Option: skip lists (or enumerate if you want per-index metrics)
        return
    else:
        # Keep only finite numbers
        if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
            out[prefix] = float(obj)

# After: rep = MLflowTracker.try_read_json(report_path)



def build_cmd(args) -> List[str]:
    """
    Build the CLI command to execute your existing script:
    'python -m src.training.run_lstm ...'
    Adjust flag names here if your run_lstm.py uses different ones.
    """
    cmd = [
        sys.executable, "-m", "src.training.run_lstm",
        "--ticker", args.ticker,
        "--interval", args.interval,
        "--start", args.start,
        "--test-start", args.test_start,
        "--horizon", str(args.horizon),
        "--seq-len", str(args.seq_len),
        "--hidden", str(args.hidden),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    if args.bidirectional:
        cmd.append("--bidirectional")
    if args.pos_weight is not None:
        cmd += ["--pos-weight", str(args.pos_weight)]
    return cmd

def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device()

    # Prepare dataset (optional pre-check so the run fails early if data is misconfigured)
    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        test_start=args.test_start,
        horizon=args.horizon
    )
    _ = prepare_dataset(cfg, seq_len=args.seq_len)

    tags = {
        "model": "LSTM",
        "ticker": args.ticker,
        "interval": args.interval,
        "horizon": str(args.horizon),
        "seq_len": str(args.seq_len),
        "device": str(device),
        "launcher": "subprocess",
    }

    with MLflowTracker(experiment_name=args.experiment, run_name=args.run_name, tags=tags) as trk:
        # Log hyperparameters upfront
        trk.log_params({
            "data": {
                "ticker": args.ticker,
                "interval": args.interval,
                "start": args.start,
                "test_start": args.test_start,
                "horizon": args.horizon,
                "seq_len": args.seq_len
            },
            "model": {
                "type": "LSTMClassifier",
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
                "bidirectional": args.bidirectional
            },
            "train": {
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "pos_weight": args.pos_weight
            },
            "env": {
                "device": str(device),
                "seed": args.seed
            }
        })

        # Run your existing training script as a subprocess
        cmd = build_cmd(args)
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Surface stdout/stderr to console for debugging and keep them in MLflow as artifacts
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)

        # Optionally log raw stdout/stderr as text artifacts
        out_log = os.path.join(ART_DIR, "run_lstm_stdout.txt")
        err_log = os.path.join(ART_DIR, "run_lstm_stderr.txt")
        os.makedirs(ART_DIR, exist_ok=True)
        with open(out_log, "w") as f:
            f.write(result.stdout or "")
        with open(err_log, "w") as f:
            f.write(result.stderr or "")
        trk.maybe_log_file(out_log, artifact_path="logs")
        trk.maybe_log_file(err_log, artifact_path="logs")

        # If training failed, mark the run and exit early (artifacts may still help diagnose)
        if result.returncode != 0:
            trk.set_tags({"run_status": "failed_subprocess"})
            return

        # Log produced artifacts (checkpoint, scaler, logs, test report)
        trk.maybe_log_file(os.path.join(ART_DIR, "lstm_classifier.pt"), artifact_path="artifacts")
        trk.maybe_log_file(os.path.join(ART_DIR, "scaler.joblib"), artifact_path="artifacts")
        trk.maybe_log_file(os.path.join(ART_DIR, "lstm_logs.csv"), artifact_path="logs")
        trk.maybe_log_file(os.path.join(ART_DIR, "lstm_test_report.json"), artifact_path="reports")

        # Optionally load test report JSON and push key metrics into MLflow
        report_path = os.path.join(ART_DIR, "lstm_test_report.json")
        rep = MLflowTracker.try_read_json(report_path)
        if isinstance(rep, dict):
            numeric = {}
            _flatten_numeric("test", rep, numeric)  # prefix everything with 'test/*'
            if numeric:
                trk.log_metrics(numeric)
        # Log figures if present
        for fig in ["loss.png", "metrics.png", "lr.png", "lstm_equity.png", "lstm_drawdown.png"]:
            pth = os.path.join(FIG_DIR, fig)
            trk.maybe_log_file(pth, artifact_path="figures")

        trk.set_tags({"run_status": "completed"})

if __name__ == "__main__":
    main()
