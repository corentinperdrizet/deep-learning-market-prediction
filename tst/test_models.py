from __future__ import annotations

import pytest
import torch

from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerTimeSeriesClassifier


@pytest.mark.parametrize("bidirectional", [False, True])
def test_lstm_classifier_output_shape(bidirectional):
    batch, seq_len, n_features = 8, 16, 5
    model = LSTMClassifier(input_dim=n_features, hidden_size=32, num_layers=2, bidirectional=bidirectional)
    x = torch.randn(batch, seq_len, n_features)
    logits = model(x)
    assert logits.shape == (batch,)


def test_lstm_classifier_gradients_flow():
    model = LSTMClassifier(input_dim=4, hidden_size=8, num_layers=1)
    x = torch.randn(4, 6, 4)
    y = torch.randint(0, 2, (4,)).float()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_transformer_classifier_output_shape(pooling):
    batch, seq_len, n_features = 6, 20, 5
    model = TransformerTimeSeriesClassifier(
        input_dim=n_features, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32, pooling=pooling
    )
    x = torch.randn(batch, seq_len, n_features)
    logits = model(x)
    assert logits.shape == (batch,)


def test_transformer_classifier_rejects_wrong_feature_dim():
    model = TransformerTimeSeriesClassifier(input_dim=5, d_model=16, n_heads=2, n_layers=1)
    x = torch.randn(2, 10, 7)  # wrong last dim
    with pytest.raises(AssertionError):
        model(x)
