from __future__ import annotations

import numpy as np
import pytest
import torch

from src.interpret.attention import day_importance_from_attention, extract_attention_weights
from src.models.transformer import TransformerTimeSeriesClassifier


def _make_model(pooling):
    model = TransformerTimeSeriesClassifier(
        input_dim=4, d_model=8, n_heads=2, n_layers=2, dim_feedforward=16, pooling=pooling
    )
    model.eval()
    return model


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_extract_attention_weights_rows_sum_to_one(pooling):
    model = _make_model(pooling)
    x = torch.randn(3, 6, 4)
    attn = extract_attention_weights(model, x)

    expected_t = 6 + 1 if pooling == "cls" else 6
    assert attn.shape == (3, expected_t, expected_t)
    row_sums = attn.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    assert (attn >= 0).all()


def test_extract_attention_weights_requires_eval_mode():
    model = _make_model("mean")
    model.train()
    with pytest.raises(RuntimeError):
        extract_attention_weights(model, torch.randn(1, 5, 4))


def test_extract_attention_deterministic_in_eval_mode():
    model = _make_model("mean")
    x = torch.randn(2, 5, 4)
    attn1 = extract_attention_weights(model, x)
    attn2 = extract_attention_weights(model, x)
    np.testing.assert_allclose(attn1, attn2)


@pytest.mark.parametrize("pooling,expected_days", [("mean", 6), ("cls", 6)])
def test_day_importance_shape(pooling, expected_days):
    model = _make_model(pooling)
    x = torch.randn(3, 6, 4)
    attn = extract_attention_weights(model, x)
    importance = day_importance_from_attention(attn, pooling)
    assert importance.shape == (3, expected_days)
