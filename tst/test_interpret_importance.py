from __future__ import annotations

import numpy as np
import pytest

from src.interpret.importance import logreg_coefficients, permutation_importance
from src.models.baselines import LogisticRegressionTabular


def _synthetic_seq_dataset(n=400, t=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, t, 3))
    # y depends only on the LAST timestep of feature 0; features 1 and 2 are pure noise.
    y = (X[:, -1, 0] > 0).astype(int)
    return X, y


def test_permutation_importance_ranks_informative_feature_highest():
    X, y = _synthetic_seq_dataset()

    def predict_fn(Xb):
        # Hand-crafted "model": sigmoid of the last-timestep feature-0 value.
        return 1.0 / (1.0 + np.exp(-Xb[:, -1, 0]))

    df = permutation_importance(
        predict_fn, X, y, feature_names=["informative", "noise_1", "noise_2"], n_repeats=5, seed=1
    )

    assert df.iloc[0]["feature"] == "informative"
    assert df.iloc[0]["importance_mean"] > df.iloc[1]["importance_mean"]
    assert df.iloc[0]["importance_mean"] > df.iloc[2]["importance_mean"]
    assert df.attrs["baseline_score"] > 0.9  # near-perfect separator


def test_permutation_importance_rejects_wrong_feature_name_count():
    X, y = _synthetic_seq_dataset()
    with pytest.raises(ValueError):
        permutation_importance(lambda X: np.full(len(X), 0.5), X, y, feature_names=["only_one"])


def test_permutation_importance_rejects_non_3d_input():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError):
        permutation_importance(lambda X: np.full(len(X), 0.5), X, np.zeros(10), feature_names=["a", "b", "c"])


def test_logreg_coefficients_matches_raw_sklearn_and_sorts_by_magnitude():
    X, y = _synthetic_seq_dataset()
    clf = LogisticRegressionTabular(pooling="last").fit(X, y)

    df = logreg_coefficients(clf.pipe_, feature_names=["informative", "noise_1", "noise_2"])

    assert set(df.columns) == {"feature", "coefficient", "abs_coefficient"}
    assert df.iloc[0]["feature"] == "informative"
    raw_coefs = clf.pipe_.named_steps["clf"].coef_.ravel()
    assert df["abs_coefficient"].max() == pytest.approx(np.abs(raw_coefs).max())


def test_logreg_coefficients_raises_on_mismatched_feature_count():
    X, y = _synthetic_seq_dataset()
    clf = LogisticRegressionTabular(pooling="flatten_last_k", k=3).fit(X, y)
    with pytest.raises(ValueError):
        logreg_coefficients(clf.pipe_, feature_names=["a", "b", "c"])
