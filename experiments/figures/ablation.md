# Ablation Study — Deep Learning Market Prediction

This document tracks controlled experiments that isolate the impact of **features**, **sequence length**, **horizon**, and **scaler** on predictive performance and tradability.

## Protocol

- **Dataset & splits**: keep the same chronological train/val/test as main runs.
- **Model**: fix a baseline (e.g., `LSTM(hidden=128,layers=2,dropout=0.2)`), unless a row explicitly states otherwise.
- **Training**: identical seeds & epochs (or early stopping), same optimizer/scheduler.
- **Calibration**: apply **Platt** or **Isotonic** on **validation**, then reuse on **test**.
- **Thresholding**: pick θ on **validation** to maximize **F1** or **Sharpe** (state which), then apply to **test**.
- **Metrics**: report **ROC-AUC**, **PR-AUC**, **F1@θ**, **Sharpe@θ** (if returns available), and note any stability issues.

---

## Summary Table (Validation → Test)

| Experiment | Change vs Base | Val ROC-AUC | Val PR-AUC | Val Best θ (obj) | Test ROC-AUC | Test PR-AUC | Test F1@θ | Test Sharpe@θ | Notes |
|---|---|---:|---:|:---:|---:|---:|---:|---:|---|
| BASE | LSTM(128x2), seq=64, horizon=1, StandardScaler, all features | 0.51x | 0.52x | 0.60 (F1) | 0.51x | 0.52x | 0.5x | 0.0x | Reference |
| -RSI | Drop RSI |  |  |  |  |  |  |  |  |
| -MACD | Drop MACD(+signal,hist) |  |  |  |  |  |  |  |  |
| seq=32 | Sequence length 32 |  |  |  |  |  |  |  |  |
| seq=96 | Sequence length 96 |  |  |  |  |  |  |  |  |
| horizon=3 | Predict t+3 |  |  |  |  |  |  |  |  |
| Robust | RobustScaler instead of Standard |  |  |  |  |  |  |  |  |
| Transformer | Encoder-only (128/4/3) |  |  |  |  |  |  |  |  |

> Fill with your numbers. Use consistent seeds; add rows for any other feature drops (e.g., volatility, calendar encodings, lags).

---

## How to Run Each Ablation

1. **Prepare data variant** (e.g., drop RSI):
   - Add a flag in `src/data/features.py` or a config switch to skip specific indicators.
2. **Train** using the same script (`run_lstm.py` or `run_transformer.py`) and fixed seed.
3. **Evaluate on validation** → save raw probabilities.
4. **Calibrate** on validation; **optimize θ** (F1 *or* Sharpe).
5. **Lock** calibrator + θ; **evaluate on test**.
6. **Log** numbers into the table above and keep artifacts under `data/artifacts/ablation/<tag>/`.

---

## Reporting Tips

- Plot reliability diagrams **before/after** calibration.
- Keep threshold curves (F1 vs θ, Sharpe vs θ) for 1–2 key variants.
- Record runtime and any instability (divergence, overfitting, high variance across seeds).
