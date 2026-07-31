# -----------------------------
# File: src/interpret/attention.py
# -----------------------------
"""Extract self-attention weights from TransformerTimeSeriesClassifier.

nn.TransformerEncoderLayer doesn't expose its internal attention weights
through the model's normal forward() (PyTorch's fast path calls self_attn
with need_weights=False for speed). To visualize "what days the model
attended to", we manually replay each pre-norm encoder layer's math,
issuing one extra self_attn(..., need_weights=True) call purely to capture
the weights, while still calling the layer's own forward() to propagate the
real hidden state -- this must run with the model in eval() mode so the
diagnostic call and the real call see identical (dropout-free) attention.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.transformer import TransformerTimeSeriesClassifier


@torch.no_grad()
def extract_attention_weights(model: TransformerTimeSeriesClassifier, x: torch.Tensor) -> np.ndarray:
    """Return per-layer-averaged attention, shape (batch, T', T') where
    T' = seq_len (+1 if pooling == "cls", for the prepended CLS token).
    """
    if model.training:
        raise RuntimeError(
            "Call model.eval() before extracting attention (dropout would make it non-deterministic)."
        )

    b = x.shape[0]
    h = model.in_proj(x)
    if model.pooling == "cls":
        cls = model.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
    h = model.pos_enc(h)

    layer_maps = []
    for layer in model.encoder.layers:
        normed = layer.norm1(h)
        _, attn_weights = layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )
        layer_maps.append(attn_weights)
        h = layer(h)  # real forward, to correctly propagate state to the next layer

    avg_attn = torch.stack(layer_maps, dim=0).mean(dim=0)  # (B, T', T')
    return avg_attn.cpu().numpy()


def day_importance_from_attention(attn: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce a (B, T', T') attention map to a per-day importance vector
    (B, T), consistent with how the model actually reads attention:
      - pooling="mean": final representation averages over all positions,
        so a day's importance is how much attention it receives, averaged
        over all query positions.
      - pooling="cls": only the CLS token's representation is classified,
        so a day's importance is how much the CLS token attended to it.
    """
    if pooling == "cls":
        return attn[:, 0, 1:]  # CLS row, excluding CLS's attention to itself
    return attn.mean(axis=1)
