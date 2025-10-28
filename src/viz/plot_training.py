# src/viz/plot_training.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_loss(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, "loss.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def plot_metrics(df: pd.DataFrame, outdir: str, smooth: int = 0):
    # Optionnel: lissage simple (rolling mean) des métriques
    work = df.copy()
    if smooth and smooth > 1:
        for col in ["val_pr_auc", "val_roc_auc", "val_f1"]:
            if col in work:
                work[col] = work[col].rolling(window=smooth, min_periods=1).mean()

    plt.figure(figsize=(8, 4.5))
    if "val_pr_auc" in work:
        plt.plot(work["epoch"], work["val_pr_auc"], label="Val PR-AUC")
    if "val_roc_auc" in work:
        plt.plot(work["epoch"], work["val_roc_auc"], label="Val ROC-AUC")
    if "val_f1" in work:
        plt.plot(work["epoch"], work["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, "metrics.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def plot_lr(df: pd.DataFrame, outdir: str):
    if "lr" not in df.columns:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate (ReduceLROnPlateau)")
    plt.tight_layout()
    outpath = os.path.join(outdir, "lr.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    ap = argparse.ArgumentParser(description="Plot training curves from lstm_logs.csv")
    ap.add_argument("--logs", type=str, default="data/artifacts/lstm_logs.csv", help="Path to CSV logs")
    ap.add_argument("--outdir", type=str, default="experiments/figures", help="Output directory for plots")
    ap.add_argument("--smooth", type=int, default=0, help="Rolling window for smoothing metrics (0=off)")
    args = ap.parse_args()

    _ensure_outdir(args.outdir)
    df = pd.read_csv(args.logs)

    # Coerce numeric cols just in case they are strings
    for c in ["epoch", "train_loss", "val_loss", "val_pr_auc", "val_roc_auc", "val_f1", "lr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    plot_loss(df, args.outdir)
    plot_metrics(df, args.outdir, smooth=args.smooth)
    plot_lr(df, args.outdir)


if __name__ == "__main__":
    main()
