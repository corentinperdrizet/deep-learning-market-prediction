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
├── 01_baselines.ipynb              # Notebook to run and analyze baseline models
├── README.md                       # Main project documentation
├── requirements.txt                 # Dependencies list
│
├── data/
│   ├── artifacts/                   # Trained models, logs, reports, and scalers
│   │   ├── baselines_metrics.csv          # Metrics summary for baseline models
│   │   ├── lstm_classifier.pt             # Best LSTM model checkpoint
│   │   ├── lstm_logs.csv                  # LSTM training log (loss, metrics per epoch)
│   │   ├── lstm_test_report.json          # Final test metrics for LSTM
│   │   ├── transformer_classifier.pt      # Best Transformer model checkpoint
│   │   ├── transformer_logs.csv           # Transformer training log
│   │   ├── transformer_test_report.json   # Final test metrics for Transformer
│   │   └── scaler.joblib                  # Saved feature scaler for reproducibility
│   ├── processed/                   # Cleaned, feature-engineered datasets
│   │   └── BTC-USD_1d_dataset.parquet
│   └── raw/                         # Original OHLCV data
│       └── BTC-USD_1d.parquet
│
├── experiments/
│   └── figures/                     # Visualization outputs (loss, metrics curves)
│       ├── loss.png
│       ├── lr.png
│       └── metrics.png
│   └── ablation.md
│
├── notebooks/                       # Jupyter notebooks for experiments and EDA
│
├── src/
│   ├── app/                         # (Planned) Streamlit dashboard for visualization
│   ├── backtest/                    # (Planned) Strategy backtesting module
│   ├── labeling/                    # (Planned) Label generation and event-based labeling
│   ├── utils/                       # (Planned) General-purpose utility functions
│
│   ├── data/                        # Data preparation and feature engineering pipeline
│   │   ├── __init__.py
│   │   ├── config.py                # Data configuration parameters
│   │   ├── config_bridge.py         # Bridge config for cross-module consistency
│   │   ├── dataset.py               # Orchestrates full dataset creation (end-to-end)
│   │   ├── features.py              # Technical indicators (RSI, MACD, volatility, etc.)
│   │   ├── loaders.py               # Data loading and cleaning (e.g., yfinance)
│   │   ├── paths.py                 # Handles data paths and directories
│   │   ├── preprocessing.py         # Label creation, merging, and feature alignment
│   │   ├── quality.py               # Data quality checks (missing, duplicates)
│   │   ├── scaling.py               # Scaler fitting and transformations
│   │   ├── sequences.py             # Rolling window sequence generation for DL
│   │   └── viz.py                   # Data visualization utilities
│
│   ├── models/                      # Model architectures (baselines and DL)
│   │   ├── baselines.py             # Buy & Hold, SMA, Logistic Regression, XGBoost (optional)
│   │   ├── lstm.py                  # LSTMClassifier implementation
│   │   ├── transformer.py           # TransformerTimeSeriesClassifier implementation
│   │   └── __init__.py
│
│   ├── training/                    # Training, evaluation, and metrics modules
│   │   ├── calibration.py           # Platt & Isotonic calibration
│   │   ├── thresholds.py            # Threshold optimization (F1/Sharpe)
│   │   ├── dataloaders.py           # Converts NumPy data into Torch DataLoaders
│   │   ├── evaluate.py              # Evaluation helpers for classification/regression
│   │   ├── metrics.py               # Metric computations (ROC-AUC, PR-AUC, F1, etc.)
│   │   ├── run_baselines.py         # Script to train and log baseline models
│   │   ├── run_lstm.py              # Script to train the LSTM model
│   │   ├── run_transformer.py       # Script to train the Transformer model
│   │   ├── trainer.py               # Training loop, validation, early stopping, checkpointing
│   │   └── utils.py                 # Device selection, seed setup, and helpers
│
│   └── viz/                         # Plotting utilities for training curves
│       └── plot_training.py         # Generates plots from training logs
│
└── tst/                             # Unit and integration tests (optional)
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

### 🧠 Why Baselines ?
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

## ⚙️ Step 4 — Transformer Time-Series Model (Encoder-Only)

### 🎯 Objective
The goal of this step was to **implement and evaluate a Temporal Transformer (encoder-only)** architecture for financial time series classification, and compare its performance against the LSTM baseline.

The Transformer is designed to handle **temporal dependencies** and **longer context windows** via self-attention, potentially capturing complex interactions between features that recurrent models might miss.

---

### 🧩 Model Architecture
Implemented in `src/models/transformer.py`, the model follows an encoder-only design inspired by *Attention Is All You Need*:

- **Feature embedding:** Linear projection from input features (F) → hidden dimension (`d_model`).
- **Positional encoding:** Sinusoidal encoding to inject temporal order.
- **Encoder stack:** `n_layers` TransformerEncoderLayers (`d_model`, `n_heads`, `dim_feedforward`, `dropout`).
- **Pooling:** Mean-pooling or CLS token pooling over the sequence dimension.
- **Classification head:** MLP projection (`Linear → GELU → Dropout → Linear → Logit`).
- **Loss:** `BCEWithLogitsLoss` for binary direction prediction (up/down).

Default hyperparameters:
`d_model=128, n_heads=4, n_layers=3, ff=256, dropout=0.1, lr=2e-4`


---

### 🧱 Implementation Details
- New training script: `src/training/run_transformer.py`
  - Compatible with the existing data pipeline and Trainer utilities.
  - Uses `AdamW` optimizer with weight decay.
  - Early stopping and learning-rate scheduling on validation `PR-AUC`.
  - All artifacts saved under `data/artifacts/`:
    - `transformer_classifier.pt` — best model checkpoint
    - `transformer_logs.csv` — per-epoch metrics
    - `transformer_test_report.json` — final test evaluation

- Reuses the same dataset and preprocessing logic as the LSTM:
`(X_train, y_train), (X_val, y_val), (X_test, y_test)
shape = (N, seq_len=64, n_features=13)`

- Both pooling strategies were tested:
- **Mean pooling** (average of hidden states)
- **CLS pooling** (learnable token prepended to the sequence)

---

### 🧪 Results
Training was **stable** across multiple runs (no exploding loss or divergence).  
The model converged with validation metrics close to the LSTM baseline.

Example test metrics (BTC-USD, daily horizon=1):

| Metric | Transformer (mean) | Transformer (CLS) | LSTM baseline |
|--------|---------------------|-------------------|----------------|
| Accuracy | 0.509 | 0.509 | 0.510 |
| F1 (pos) | 0.675 | 0.675 | 0.674 |
| ROC-AUC | 0.517 | 0.510 | 0.512 |
| PR-AUC | 0.526 | 0.520 | 0.529 |
| Brier | 0.2499 | 0.2499 | 0.2498 |

---

### 📈 Interpretation
- The Transformer achieves **comparable performance** to the LSTM, confirming that the training pipeline, feature engineering, and label construction are sound.
- Both models show a **weak but non-random predictive signal** (ROC-AUC slightly above 0.5) at the 1-day horizon — consistent with market efficiency.
- The CLS pooling version performs slightly worse, likely due to limited data size and short sequence lengths.

---

### 🚀 Next Improvements
- **Longer context**: try `--seq-len 96` or `128`.
- **Larger model**: increase `n_layers` to 4–6 and `ff` to 512.
- **Better features**: add lagged returns, rolling percentiles, volatility regime indicators.
- **Alternative encodings**: implement `Time2Vec` or learnable positional embeddings.
- **Cosine scheduler with warmup** for smoother optimization.
- **Multi-asset training** (BTC + ETH + S&P500) with ticker embeddings.

---

### ✅ Step Outcome
✔️ Transformer encoder-only model implemented and trained successfully.  
✔️ Training stable, metrics on par with LSTM baseline.  
✔️ Ready for further experimentation with richer features and multi-asset setups.

---

# ⚙️ Step 5 — Calibration, Thresholds, and Ablation

## 🎯 Objective
This step focuses on transforming raw predictive scores into **actionable trading signals**.  
Even if a model has a decent ROC-AUC or PR-AUC, its probabilities might not correspond to real-world likelihoods, and its decision threshold (default 0.5) might not be optimal.  
The calibration and thresholding phase ensures that the model's outputs can be reliably interpreted and used in a backtesting environment.

---

## 🧩 Implemented Modules

### 1️⃣ Calibration

Two calibration methods were implemented in `src/training/calibration.py`:

- **Platt Scaling** — a parametric sigmoid-based calibration that maps raw probabilities to better-calibrated outputs.  
- **Isotonic Regression** — a non-parametric, monotonic calibration model that can adapt to arbitrary probability distortions.

Each method provides:
- `fit(y_val, p_val)` — learns the calibration mapping on validation data.  
- `transform(p)` — applies the learned mapping to new probabilities.  
- `save(path)` / `load(path)` — persist the calibrator for later reuse.

Additionally, the module includes:
- `expected_calibration_error()` and `calibration_report()` — compute ECE, MCE, and Brier Score.  
- `plot_reliability_diagram()` — plots calibration curves (empirical accuracy vs. confidence).

This allows us to visualize how well a model's probabilities align with true event frequencies.

---

### 2️⃣ Threshold Optimization

The module `src/training/thresholds.py` introduces a **systematic search** for the optimal decision threshold θ.  
Rather than defaulting to θ=0.5, this step finds the value that maximizes a given objective on the validation set:

- **F1** — for pure classification tasks (maximizing accuracy and recall balance).  
- **Sharpe Ratio** — when next-step returns are available, to directly optimize the risk-adjusted profitability of the signal.

Main components:
- `grid_search_threshold()` — evaluates metrics (Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC, Sharpe) across thresholds.  
- `plot_metric_vs_threshold()` — visualizes how performance changes with θ.  
- `ThresholdSearchResult` — dataclass storing the best θ, best value, and the full metrics table.

---

### 3️⃣ Ablation Protocol

A dedicated file `experiments/ablation.md` was added to standardize controlled experiments and feature-importance studies.  
It defines how to test the impact of:
- removing or modifying specific features (e.g., RSI, MACD),
- changing sequence length (32, 64, 96),
- altering prediction horizon (1, 3, 7 days),
- using different scalers (Standard vs. Robust).

Each experiment reports:
- Validation metrics (ROC-AUC, PR-AUC, F1@θ)
- Test metrics (ROC-AUC, PR-AUC, F1@θ, Sharpe@θ)
- Notes on stability, convergence, and runtime.

---

## 🧪 Usage Example

Calibration and thresholding are applied **after** model training, using validation outputs:

```python
from src.training.calibration import PlattCalibrator, calibration_report, plot_reliability_diagram
from src.training.thresholds import grid_search_threshold, plot_metric_vs_threshold

# 1. Fit calibrator on validation
cal = PlattCalibrator().fit(y_val, p_val)
p_val_cal = cal.transform(p_val)

# 2. Evaluate calibration
print(calibration_report(y_val, p_val))
plot_reliability_diagram(y_val, p_val_cal)

# 3. Optimize threshold (e.g., for Sharpe)
res = grid_search_threshold(y_val, p_val_cal, objective="sharpe", returns_next=returns_val)
print("Best θ:", res.best_threshold)
plot_metric_vs_threshold(res.table, metric="sharpe")
```

Once calibration and optimal θ are determined on validation, they are **frozen** and reused on the test set without retraining.

---

## ✅ Step Outcome
✔️ Implemented Platt and Isotonic calibration modules.  
✔️ Added threshold optimization with F1 and Sharpe objectives.  
✔️ Created standardized ablation protocol for systematic testing.  
✔️ Ready for integration into the backtesting and evaluation pipeline (Step 6).

---

## 🧠 Key Takeaway
This step bridges the gap between *predictive accuracy* and *trading usability*.  
A well-calibrated, threshold-optimized model provides interpretable, consistent probabilities that can be translated into real trading actions.

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
