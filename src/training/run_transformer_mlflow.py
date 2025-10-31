import argparse
import os
import subprocess
import sys
from typing import Dict, Any, List

from src.data.dataset import prepare_dataset
from src.data.config import DataConfig
from src.training.utils import select_device, set_seed
from src.track.mlflow_utils import MLflowTracker

ART_DIR = "data/artifacts"
FIG_DIR = "experiments/figures"

def parse_args():
    p = argparse.ArgumentParser(description="Transformer run with MLflow tracking (subprocess mode)")

    # ----------------- Data -----------------
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)

    # ----------------- Model -----------------
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")

    # ----------------- Train -----------------
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # ----------------- Misc -----------------
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--experiment", default="deep-learning-market-prediction")
    p.add_argument("--run-name", default=None)
    return p.parse_args()

def build_cmd(args) -> List[str]:
    """
    Build the CLI command to execute your existing Transformer script.
    Adjust flag names to match src.training.run_transformer if needed.
    """
    cmd = [
        sys.executable, "-m", "src.training.run_transformer",
        "--ticker", args.ticker,
        "--interval", args.interval,
        "--start", args.start,
        "--test-start", args.test_start,
        "--horizon", str(args.horizon),
        "--seq-len", str(args.seq_len),
        "--d-model", str(args.d_model),
        "--n-heads", str(args.n_heads),
        "--n-layers", str(args.n_layers),
        "--ff", str(args.ff),
        "--dropout", str(args.dropout),
        "--pooling", args.pooling,
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed),
    ]
    return cmd

def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device()

    cfg = DataConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        test_start=args.test_start,
        horizon=args.horizon
    )
    _ = prepare_dataset(cfg, seq_len=args.seq_len)

    tags = {
        "model": "TransformerEncoder",
        "ticker": args.ticker,
        "interval": args.interval,
        "horizon": str(args.horizon),
        "seq_len": str(args.seq_len),
        "device": str(device),
        "pooling": args.pooling,
        "launcher": "subprocess",
    }

    with MLflowTracker(experiment_name=args.experiment, run_name=args.run_name, tags=tags) as trk:
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
                "type": "TransformerTimeSeriesClassifier",
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "ff": args.ff,
                "dropout": args.dropout,
                "pooling": args.pooling
            },
            "train": {
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay
            },
            "env": {
                "device": str(device),
                "seed": args.seed
            }
        })

        cmd = build_cmd(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)

        out_log = os.path.join(ART_DIR, "run_transformer_stdout.txt")
        err_log = os.path.join(ART_DIR, "run_transformer_stderr.txt")
        os.makedirs(ART_DIR, exist_ok=True)
        with open(out_log, "w") as f:
            f.write(result.stdout or "")
        with open(err_log, "w") as f:
            f.write(result.stderr or "")
        trk.maybe_log_file(out_log, artifact_path="logs")
        trk.maybe_log_file(err_log, artifact_path="logs")

        if result.returncode != 0:
            trk.set_tags({"run_status": "failed_subprocess"})
            return

        # Log expected artifacts
        trk.maybe_log_file(os.path.join(ART_DIR, "transformer_classifier.pt"), artifact_path="artifacts")
        trk.maybe_log_file(os.path.join(ART_DIR, "scaler.joblib"), artifact_path="artifacts")
        trk.maybe_log_file(os.path.join(ART_DIR, "transformer_logs.csv"), artifact_path="logs")
        trk.maybe_log_file(os.path.join(ART_DIR, "transformer_test_report.json"), artifact_path="reports")

        # Log test metrics if a JSON report was produced
        rep_path = os.path.join(ART_DIR, "transformer_test_report.json")
        rep = MLflowTracker.try_read_json(rep_path)
        if isinstance(rep, dict):
            trk.log_metrics({f"test/{k}": v for k, v in rep.items()})

        for fig in ["loss.png", "metrics.png", "lr.png", "lstm_equity.png", "lstm_drawdown.png"]:
            pth = os.path.join(FIG_DIR, fig)
            trk.maybe_log_file(pth, artifact_path="figures")

        trk.set_tags({"run_status": "completed"})

if __name__ == "__main__":
    main()
