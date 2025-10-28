# Deep Learning Market Prediction

## 🧭 Project Overview
This project aims to **predict financial market movements** using **deep learning models** such as LSTM, GRU, and Temporal Transformers.  
The goal is to forecast the **direction or return** of an asset (e.g., BTC/USD, ETH/USD, S&P500) over short time horizons and evaluate the resulting strategies through backtesting.

The project demonstrates:
- Advanced understanding of **time series modeling**
- Application of **machine learning in finance**
- Implementation of a **complete end-to-end ML pipeline** (data → model → evaluation → dashboard)

---

## 🧱 Project Structure

```
deep-learning-market-prediction/
├── data/
│   ├── raw/            # Raw downloaded OHLCV data
│   ├── processed/      # Cleaned, feature-engineered datasets
│   └── artifacts/      # Saved scalers, trained models, etc.
├── src/
│   ├── data/           # Data processing module (fully functional)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── loaders.py
│   │   ├── quality.py
│   │   ├── features.py
│   │   ├── preprocessing.py
│   │   ├── scaling.py
│   │   ├── sequences.py
│   │   ├── viz.py
│   │   └── dataset.py
│   ├── models/         # LSTM, Transformer, Baselines (to be implemented)
│   ├── training/       # Training loops, metrics, evaluation (to be added)
│   ├── backtest/       # Strategy simulation and performance metrics (to be added)
│   └── app/            # Streamlit dashboard (to be added)
└── README.md           # Project documentation
```

---

## 📊 Data Pipeline (Completed)

### 1️⃣ Objective
The data pipeline is designed to transform raw market data into **clean, structured, and model-ready sequences** suitable for deep learning architectures.

It ensures:
- No data leakage (strict temporal logic)
- Modular and reproducible workflow
- Support for multiple assets, frequencies, and label types

---

### 2️⃣ Data Flow Overview

```
(raw OHLCV data)
      │
      ▼
  loaders.py  →  Download & clean data (BTC-USD daily)
      │
      ▼
  quality.py  →  Check missing values, duplicates, gaps
      │
      ▼
  features.py  →  Compute technical indicators (returns, RSI, MACD, etc.)
      │
      ▼
  preprocessing.py  →  Combine features, create labels, split train/val/test
      │
      ▼
  scaling.py  →  Fit scaler (train only), transform features safely
      │
      ▼
  sequences.py  →  Create rolling sequences for DL models (e.g., 64-day windows)
      │
      ▼
  dataset.py  →  Orchestrate full process & save outputs
```

---

### 3️⃣ Core Features
Each sample (day) is described by 13 core features derived from historical prices:

| Category | Feature | Description |
|-----------|----------|-------------|
| Returns | `log_ret` | Logarithmic daily return |
| Volatility | `vol_20` | 20-day rolling standard deviation of log returns |
| Momentum | `rsi_14` | 14-day Relative Strength Index |
| Trend | `macd`, `macd_signal`, `macd_hist` | MACD indicators (12/26/9) |
| Multi-horizon returns | `ret_1`, `ret_3`, `ret_7`, `ret_14` | Returns over multiple past horizons |
| Calendar | `dow`, `dow_sin`, `dow_cos` | Day of week and cyclic encoding |

---

### 4️⃣ Labeling
The model can target either **direction** or **return** prediction.

- **Direction (classification):**
  `python
  y_t = 1 if log(P_{t+1}/P_t) > 0 else 0
  `python
- **Return (regression):**
  `python
  y_t = log(P_{t+h}/P_t)
  `python

For now, the pipeline uses **`direction`** with a **1-day horizon**, i.e., predicting whether BTC/USD will go up or down tomorrow.

---

### 5️⃣ Time-based Splitting
Data is split chronologically into three parts:

| Split | Description | Example period |
|--------|--------------|----------------|
| Train | Used to fit model & scaler | 2018–2022 mid |
| Validation | Used for hyperparameter tuning | 2022 mid–2022 end |
| Test | Out-of-sample evaluation | 2023–present |

If `val_start` is not explicitly provided, 10% of the pre-test data is automatically allocated to validation.

---

### 6️⃣ Scaling
- **StandardScaler** (or optionally RobustScaler) is fit **only on the training data**.
- Applied consistently to val/test sets to avoid information leakage.
- The fitted scaler is saved to `data/artifacts/scaler.joblib` for reproducibility.

---

### 7️⃣ Sequence Construction
To feed temporal models like LSTM or Transformers, the data is transformed into sliding windows:

```
Sequence length (seq_len) = 64 days
Features per step = 13
```

This yields arrays like:

```
X_train: (1561, 64, 13)
y_train: (1561,)
X_val:   (117, 64, 13)
y_val:   (117,)
X_test:  (968, 64, 13)
y_test:  (968,)
```

Each training sample contains the past 64 days of market features, and the target is the **next-day direction**.

---

### 8️⃣ Data Quality
The pipeline performs several quality checks before processing:
- Missing values per column
- Duplicate timestamps
- Missing calendar days (for crypto or traditional assets)

It reports these in the console to ensure transparency and reproducibility.

---

### 9️⃣ Output Summary
After execution (`python -m src.data.dataset`), the function `prepare_dataset()` returns a dictionary:

| Key | Type | Description |
|-----|------|--------------|
| `X_train`, `X_val`, `X_test` | `np.ndarray` | Time-series sequences |
| `y_train`, `y_val`, `y_test` | `np.ndarray` | Corresponding labels |
| `features` | `list` | Feature names |
| `idx` | `dict` | Time index ranges for each split |
| `meta` | `dict` | Dataset metadata (ticker, horizon, etc.) |

Example:
`python
{
  "X_train": (1561, 64, 13),
  "y_train": (1561,),
  "features": ["log_ret", "vol_20", "rsi_14", ...],
  "meta": {"ticker": "BTC-USD", "interval": "1d", "label_type": "direction"}
}
`python

---

## 🚀 Next Steps
- [ ] Implement **LSTM and Transformer** architectures (`src/models/`)
- [ ] Add **training loops & metrics** (`src/training/`)
- [ ] Develop **backtesting module** for strategy evaluation (`src/backtest/`)
- [ ] Build a **Streamlit dashboard** to visualize results interactively (`src/app/`)
- [ ] Extend dataset support to multiple assets and timeframes (ETH, S&P500, etc.)

---

## 🧩 References
- Yahoo Finance API (`yfinance`)
- pandas, NumPy, scikit-learn
- Technical Analysis concepts (RSI, MACD, rolling volatility)
- Time series forecasting and deep learning best practices

---

## 📁 Notes
This project is modular by design:
- Each submodule (`src/data`, `src/models`, `src/training`, etc.) can be developed and tested independently.
- Config-driven approach ensures flexibility for future assets and modeling techniques.
- Fully deterministic (fixed random seed, cached data, reproducible splits).

---
