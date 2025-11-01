# src/app/streamlit_app.py

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Robust import: works both as "script" and "package" ---
try:
    # When executed as a package (e.g., `python -m src.app.streamlit_app`)
    from .utils import (
        discover_available_runs,
        load_json_if_exists,
        load_csv_if_exists,
        find_figure_if_exists,
        find_processed_parquet,
        load_parquet_head_if_exists,
        compute_confusion_matrix,
    )
except Exception:
    # When executed as a script (e.g., `streamlit run src/app/streamlit_app.py`)
    import sys
    import pathlib

    # Add <repo>/src to sys.path so "from app.utils" can be imported
    SRC_DIR = pathlib.Path(__file__).resolve().parents[1]  # .../src
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from app.utils import (  # type: ignore
        discover_available_runs,
        load_json_if_exists,
        load_csv_if_exists,
        find_figure_if_exists,
        find_processed_parquet,
        load_parquet_head_if_exists,
        compute_confusion_matrix,
    )

# ---------- Page config ----------
st.set_page_config(
    page_title="DL Market Prediction — Dashboard",
    layout="wide",
)

st.title("📊 Deep Learning Market Prediction — Dashboard")

# ---------- Discover runs ----------
ARTIFACTS_DIR = Path("data/artifacts")
FIGURES_DIR = Path("experiments/figures")
PROCESSED_DIR = Path("data/processed")

available = discover_available_runs(ARTIFACTS_DIR, FIGURES_DIR)
if not available:
    st.warning("No run detected. Make sure you have artifacts in `data/artifacts/` and figures in `experiments/figures/`.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")

run_names = list(available.keys())
selected_run = st.sidebar.selectbox("Run / Model", run_names, index=0)

theta = st.sidebar.slider("Threshold θ (classification)", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
fees_bps = st.sidebar.number_input("Fees (bps)", min_value=0, max_value=100, value=10, step=1)
st.sidebar.caption("θ is used in **Signals** tab if probabilities are available.\nFees are displayed for context in Backtest.")

run_info = available[selected_run]

# ---------- Load report JSON (metrics + hyperparams) ----------
report_json = load_json_if_exists(run_info.get("test_report"))
if report_json:
    run_info["hyperparams"] = report_json.get("hyperparams") or report_json.get("config") or {}
    run_info["test_metrics"] = report_json.get("test_metrics") or report_json.get("metrics") or {}

# ---------- Tabs ----------
tab_models, tab_backtest, tab_signals, tab_data = st.tabs(["Models", "Backtest", "Signals", "Data"])

# =========================================
# TAB: Models
# =========================================
with tab_models:
    st.subheader("🧠 Model — Hyperparameters & Test Metrics")

    col_hp, col_mt = st.columns(2)

    with col_hp:
        st.markdown("**Hyperparameters**")
        if report_json and run_info.get("hyperparams"):
            st.json(run_info["hyperparams"])
        else:
            st.info("No hyperparameters found in the report (`*_test_report.json`).")

    with col_mt:
        st.markdown("**Test Metrics**")
        if report_json and run_info.get("test_metrics"):
            st.json(run_info["test_metrics"])
        else:
            st.info("No test metrics found in the report (`*_test_report.json`).")

    st.markdown("---")
    st.subheader("📈 Training Curves")

    figs = {
        "Loss": find_figure_if_exists(FIGURES_DIR, run_info.get("loss_fig")),
        "Metrics (PR/ROC/F1)": find_figure_if_exists(FIGURES_DIR, run_info.get("metrics_fig")),
        "Learning Rate": find_figure_if_exists(FIGURES_DIR, run_info.get("lr_fig")),
    }

    col1, col2, col3 = st.columns(3)
    for (title, path), col in zip(figs.items(), [col1, col2, col3]):
        with col:
            st.markdown(f"**{title}**")
            if path and path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info("Figure not found.")

# =========================================
# TAB: Backtest
# =========================================
with tab_backtest:
    st.subheader("💹 Backtest — Equity & KPIs")

    eq_path = find_figure_if_exists(FIGURES_DIR, run_info.get("equity_fig"))
    dd_path = find_figure_if_exists(FIGURES_DIR, run_info.get("dd_fig"))

    col_eq, col_dd = st.columns(2)
    with col_eq:
        st.markdown("**Equity Curve**")
        if eq_path and eq_path.exists():
            st.image(str(eq_path), use_column_width=True)
        else:
            st.info("No equity curve found (e.g., `lstm_equity.png`).")

    with col_dd:
        st.markdown("**Drawdown**")
        if dd_path and dd_path.exists():
            st.image(str(dd_path), use_column_width=True)
        else:
            st.info("No drawdown figure found (e.g., `lstm_drawdown.png`).")

    st.markdown("---")
    st.markdown(f"**Displayed backtest params** — θ={theta:.2f}, fees={fees_bps} bps.")

    kpi_csv = run_info.get("kpis_csv")
    df_kpis = load_csv_if_exists(kpi_csv) if kpi_csv else None

    st.subheader("📊 KPI (test)")
    if df_kpis is not None and not df_kpis.empty:
        st.dataframe(df_kpis, use_container_width=True)
    else:
        st.info("No KPI CSV detected (e.g., `data/artifacts/lstm_backtest_kpis.csv`).")

# =========================================
# TAB: Signals
# =========================================
with tab_signals:
    st.subheader("🔔 Signals — Probability vs Price & Confusion Matrix")

    sig_csv = run_info.get("signals_csv")
    df_sig = load_csv_if_exists(sig_csv) if sig_csv else None

    if df_sig is None or df_sig.empty:
        st.info(
            "No signal file detected. Optionally provide a CSV "
            "(e.g., `data/artifacts/lstm_signals.csv`) with columns: "
            "`timestamp`, `price`, `p_up`, `y_true` (optional for confusion)."
        )
    else:
        missing = [c for c in ["timestamp", "price", "p_up"] if c not in df_sig.columns]
        if missing:
            st.warning(f"Missing columns in signals CSV: {missing}")
        else:
            try:
                df_sig = df_sig.sort_values("timestamp")
                df_sig["timestamp"] = pd.to_datetime(df_sig["timestamp"])
            except Exception:
                pass

            st.markdown("**Probability (p_up) vs Price**")
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(df_sig["timestamp"], df_sig["price"], label="Price")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")

            ax2 = ax1.twinx()
            ax2.plot(df_sig["timestamp"], df_sig["p_up"], label="p_up", alpha=0.6)
            ax2.axhline(theta, linestyle="--", linewidth=1)
            ax2.set_ylabel("p_up")

            plt.title("Price & p_up over time (θ shown)")
            st.pyplot(fig)

            if "y_true" in df_sig.columns:
                y_hat = (df_sig["p_up"].values >= theta).astype(int)
                y_true = df_sig["y_true"].astype(int).values
                cm_df, summary = compute_confusion_matrix(y_true, y_hat)
                st.markdown("**Confusion Matrix**")
                st.dataframe(cm_df, use_container_width=True)
                st.caption(summary)
            else:
                st.info("`y_true` not present — confusion matrix not computed.")

# =========================================
# TAB: Data
# =========================================
with tab_data:
    st.subheader("🧾 Data — Sample & Correlation")

    pqt = find_processed_parquet(Path("data/processed"))
    head_df = load_parquet_head_if_exists(pqt, n=500)

    if head_df is None or head_df.empty:
        st.info("No parquet found in `data/processed/` (e.g., `BTC-USD_1d_dataset.parquet`).")
    else:
        st.markdown("**Preview (up to first 500 rows)**")
        st.dataframe(head_df.head(50), use_container_width=True)

        num_df = head_df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            st.markdown("---")
            st.markdown("**Correlation (numeric features)**")
            corr = num_df.corr(numeric_only=True)

            fig2, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr.values)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            ax.set_title("Feature Correlation")
            fig2.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig2)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

st.caption("Tip: run `export PYTHONPATH=$(pwd)/src && streamlit run src/app/streamlit_app.py` if you prefer absolute imports.")
