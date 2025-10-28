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

### Implemented
- Baselines (`src/models/baselines.py`)
  - Buy & Hold, SMA(50/200), Logistic Regression, (XGBoost optional)
- Deep Learning
  - `LSTMClassifier` (`src/models/lstm.py`): multi-layer LSTM with dropout and optional bidirectionality; binary logit output for next-day direction

### Training & Evaluation
- `src/training/run_baselines.py`: trains/evaluates baselines and writes `data/artifacts/baselines_metrics.csv`
- `src/training/run_lstm.py`: end-to-end LSTM training with early stopping and test evaluation
- `src/training/trainer.py`: training loop, validation, scheduler, checkpoints, CSV logs
- `src/training/dataloaders.py`: numpy → TensorDataset/DataLoader (MPS-aware pin_memory)
- `src/training/metrics.py` / `evaluate.py`: common metrics/eval helpers

### Data
- `src/data/dataset.py`: from raw OHLCV to model-ready sequences (no leakage)
- `src/data/{config,features,preprocessing,scaling,sequences,quality,loaders,paths}.py`

### Visualization
- `src/viz/plot_training.py`: exports loss/metrics/lr plots from `data/artifacts/lstm_logs.csv` to `experiments/figures/`

### App & Backtesting (to come)
- `src/app/`: Streamlit dashboard to visualize signals, metrics and PnL curves
- `src/backtest/`: execution rules, simple strategy sim, cum. returns, Sharpe, drawdown

### Current Repo Layout (excerpt)
```
deep-learning-market-prediction/
├── data/
│   ├── artifacts/
│   │   ├── baselines_metrics.csv
│   │   ├── lstm_classifier.pt
│   │   ├── lstm_logs.csv
│   │   ├── lstm_test_report.json
│   │   └── scaler.joblib
│   ├── processed/
│   │   └── BTC-USD_1d_dataset.parquet
│   └── raw/
│       └── BTC-USD_1d.parquet
├── experiments/
│   └── figures/
│       ├── loss.png
│       ├── lr.png
│       └── metrics.png
├── src/
│   ├── data/
│   │   ├── config.py
│   │   ├── config_bridge.py
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── loaders.py
│   │   ├── paths.py
│   │   ├── preprocessing.py
│   │   ├── quality.py
│   │   ├── scaling.py
│   │   ├── sequences.py
│   │   └── viz.py
│   ├── models/
│   │   ├── baselines.py
│   │   └── lstm.py
│   ├── training/
│   │   ├── dataloaders.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── run_baselines.py
│   │   ├── run_lstm.py
│   │   ├── trainer.py
│   │   └── utils.py
│   └── viz/
│       └── plot_training.py
└── notebooks/, app/, backtest/, labeling/, utils/, tst/
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
```python
{
  "X_train": (1561, 64, 13),
  "y_train": (1561,),
  "features": ["log_ret", "vol_20", "rsi_14", ...],
  "meta": {"ticker": "BTC-USD", "interval": "1d", "label_type": "direction"}
}
```

---
## ⚙️ Step 2 — Baseline Models

### 🎯 Objective
Before training deep learning models, we first establish **simple baseline models** to understand how much predictive power can be achieved without complex architectures.  
These baselines act as **reference points** — they show what level of accuracy we can get using traditional or rule-based methods.

---

### 🧠 Why Baselines `
In financial forecasting, especially for short-term movements, markets are very noisy.  
By testing simple models first, we can verify:
- that our **data pipeline** and **labels** are correct,
- that the task is **not trivially random**,  
- and later, if a deep learning model truly brings an **improvement** beyond these simple references.

---

### 🧱 Implemented Models
The following baseline models estimate the **probability that the price goes up** (class 1) on the next step:

| Model | Description |
|--------|-------------|
| **Buy & Hold** | Predicts the base rate (average probability of upward moves). Serves as a naive benchmark. |
| **SMA Crossover (50/200)** | Rule-based signal: predicts `up` when the 50-day moving average is above the 200-day average. |
| **Logistic Regression** | Simple statistical model trained on tabular features (returns, RSI, MACD, etc.). |
| **XGBoost (optional)** | Gradient boosting classifier on flattened historical features; adds nonlinear relationships. |

---

### 🧪 Evaluation
All baselines are trained on the **training set** and evaluated on **validation** and **test** splits to measure generalization.  
Metrics include:

- Accuracy and F1-score (directional correctness)
- ROC-AUC and PR-AUC (probabilistic discrimination)
- Brier score (probability calibration)

Results are saved in `data/artifacts/baselines_metrics.csv`.

---

### 📈 Interpretation
If the baselines reach around 0.50 ROC-AUC (random-like), it means the market is difficult to predict at that horizon.  
Any future deep learning model (LSTM, Transformer) should aim to **outperform these baselines** to demonstrate added predictive value.

---

# ⚙️ Step 3 — First DL Model (LSTM/GRU)

## 🎯 Goal
Validate an end-to-end sequential deep model that takes sliding windows of features and predicts the next-day direction (binary classification). This step adds a production-style training loop with early stopping, metrics, checkpoints and plots.

---

## 🧩 What was implemented

- Models
  - `LSTMClassifier` in `src/models/lstm.py`  
    - Inputs: (batch, seq_len, n_features)  
    - Output: logits (1-dim), trained with `BCEWithLogitsLoss`  
    - Default: hidden=128, layers=2, dropout=0.2, optional bidirectional

- Data → Torch
  - `src/training/dataloaders.py` converts numpy arrays to `TensorDataset` and `DataLoader`
  - No temporal shuffling within sequences; batch-level shuffling is allowed

- Training utilities
  - `src/training/utils.py`  
    - Global seed  
    - Device selection with Apple GPU (MPS) support: prefers CUDA → MPS → CPU  
    - Checkpoint helpers

- Trainer
  - `src/training/trainer.py`  
    - train_one_epoch / validate_one_epoch  
    - Metrics: PR-AUC (Average Precision), ROC-AUC, F1  
    - Early stopping on validation PR-AUC by default  
    - ReduceLROnPlateau scheduler (no `verbose` for full torch compatibility)  
    - Artifacts:
      - Best checkpoint → `data/artifacts/lstm_classifier.pt`
      - Logs per epoch → `data/artifacts/lstm_logs.csv` (train/val loss, PR-AUC, ROC-AUC, F1, LR)

- Runner
  - `src/training/run_lstm.py`  
    - Builds a `DataConfig` and calls `prepare_dataset(cfg, seq_len)`  
    - CLI overrides: `--ticker`, `--interval`, `--start`, `--test-start`, `--horizon`, `--seq-len`, `--hidden`, `--layers`, `--dropout`, `--batch`, `--epochs`, `--lr`, `--pos-weight` etc.  
    - Tests the best checkpoint on the test split and writes `data/artifacts/lstm_test_report.json`

- Plots
  - `src/viz/plot_training.py`  
    - Generates `experiments/figures/loss.png`, `metrics.png`, `lr.png` from `lstm_logs.csv`  
    - Optional smoothing via `--smooth 3`

---

## ▶️ How to run

- Train LSTM with defaults:
  - `python -m src.training.run_lstm`

- Override example:
  - `python -m src.training.run_lstm --ticker BTC-USD --interval 1d --start 2018-01-01 --test-start 2023-01-01 --horizon 1 --seq-len 64 --hidden 128 --layers 2 --dropout 0.2 --batch 256 --epochs 30 --lr 1e-3`

- Make plots:
  - `python -m src.viz.plot_training --logs data/artifacts/lstm_logs.csv --outdir experiments/figures --smooth 3`

Artifacts are written to:
- `data/artifacts/lstm_classifier.pt`
- `data/artifacts/lstm_logs.csv`
- `data/artifacts/lstm_test_report.json`
- `experiments/figures/*.png`

---

## 🧪 Example result (your run)

- Early stopping on validation PR-AUC  
- Test set: PR-AUC ≈ 0.529, ROC-AUC ≈ 0.512  
- Interpretation: weak but non-zero predictive signal at 1-day horizon, to be compared against baselines on the same splits.

---

## 📈 Tips to improve

- Class imbalance: set `--pos-weight` if class 1 is rarer  
- Try `--bidirectional` and longer `--seq-len` (e.g., 96 or 128)  
- Feature engineering: add lags/rolling percentiles, etc.  
- Seed sweep (3–5 runs) to stabilize metrics

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
