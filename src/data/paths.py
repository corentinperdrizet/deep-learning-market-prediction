# -----------------------------
# File: src/data/paths.py
# -----------------------------
from __future__ import annotations
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return project root by searching upwards for a marker (e.g., .git or pyproject.toml).
    Falls back to current working directory if not found.
    """
    start = start or Path.cwd()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return start


def scaler_filename(ticker: str) -> str:
    """Filename for a ticker's fitted scaler.

    BTC-USD keeps the original unprefixed "scaler.joblib" for backward
    compatibility; other tickers get a ticker-suffixed name. Without this,
    training on a second ticker would silently overwrite the first ticker's
    scaler (same file for every ticker), corrupting inference for any model
    served afterwards -- a real bug found while wiring up multi-asset
    training against the serving layer.
    """
    if ticker == "BTC-USD":
        return "scaler.joblib"
    safe = ticker.replace("^", "").replace("/", "-")
    return f"scaler_{safe}.joblib"


def artifact_prefix(model_kind: str, ticker: str) -> str:
    """Filename prefix for a model's artifacts (checkpoint/logs/report).

    BTC-USD keeps unprefixed filenames (e.g. "lstm_classifier.pt") for
    backward compatibility with the dashboard's default run; every other
    ticker gets a ticker suffix so multi-asset runs never overwrite each
    other's artifacts.
    """
    if ticker == "BTC-USD":
        return model_kind
    safe = ticker.replace("^", "").replace("/", "-")
    return f"{model_kind}_{safe}"


def data_dir() -> Path:
    root = project_root()
    d = root / "data"
    d.mkdir(exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    (d / "processed").mkdir(exist_ok=True)
    (d / "artifacts").mkdir(exist_ok=True)
    return d


