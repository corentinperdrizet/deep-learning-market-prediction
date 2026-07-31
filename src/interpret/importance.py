# -----------------------------
# File: src/interpret/importance.py
# -----------------------------
"""Model-agnostic interpretability for sequence models + a direct-coefficient
view for the LogisticRegression baseline.

Deliberately avoids adding SHAP as a dependency (not in this project's
environment, and a nontrivial ~200MB install for what permutation importance
already covers for a tabular/sequence binary classifier): permutation
importance needs no library beyond numpy/sklearn and works identically for
the LSTM, the Transformer, or any sklearn baseline through one uniform
`predict_fn(X) -> proba` signature.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def torch_predict_fn(model: torch.nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a trained torch classifier as predict_fn(X: (N,T,F)) -> proba (N,)."""

    @torch.no_grad()
    def _predict(X: np.ndarray) -> np.ndarray:
        model.eval()
        logits = model(torch.from_numpy(X).float().to(device))
        return 1.0 / (1.0 + np.exp(-logits.cpu().numpy()))

    return _predict


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """For each feature, shuffle it across samples (all timesteps moved
    together, so within-sample temporal structure of every OTHER feature
    stays intact) and measure the drop in `metric`. A larger mean drop means
    the model relies on that feature more.

    X: (N, T, F). Returns a DataFrame sorted by importance_mean descending.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    n, _t, f = X.shape
    if len(feature_names) != f:
        raise ValueError(f"feature_names has {len(feature_names)} entries but X has {f} features")

    rng = np.random.default_rng(seed)
    baseline = metric(y, predict_fn(X))

    rows = []
    for feat_idx, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            X_perm = X.copy()
            X_perm[:, :, feat_idx] = X[perm, :, feat_idx]
            score_perm = metric(y, predict_fn(X_perm))
            drops.append(baseline - score_perm)
        rows.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops, ddof=1)),
            }
        )

    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df.attrs["baseline_score"] = baseline
    return df


def logreg_coefficients(pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Direct coefficient view for a fitted LogisticRegressionTabular.pipe_
    (sklearn Pipeline with a "clf" LogisticRegression step). Only meaningful
    for pooling="last"/"mean" (one coefficient per raw feature); raises for
    "flatten_last_k" where coefficients correspond to (timestep, feature)
    pairs rather than features alone.
    """
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_.ravel()
    if len(coefs) != len(feature_names):
        raise ValueError(
            f"Got {len(coefs)} coefficients but {len(feature_names)} feature names -- "
            "this pipeline was likely fit with pooling='flatten_last_k', which doesn't "
            "map 1:1 to feature_names."
        )
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
