# src/models/transformer.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al.) for [B, T, D] tensors.
    Registers a buffer PE[T, D] and adds it to the input embedding.
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TransformerTimeSeriesClassifier(nn.Module):
    """
    Encoder-only Transformer for next-step direction classification.

    Input:  [B, T, F]  (sequence of tabular features)
    Steps:  Linear(F->D) -> +PE -> TransformerEncoder -> Temporal pooling -> Head -> Logit
    Output: [B] logits (use BCEWithLogitsLoss)
    """
    def __init__(
        self,
        input_dim: int,          # F
        d_model: int = 128,      # D
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048,
        layer_norm_final: bool = True,
        pooling: str = "mean",   # "mean" | "cls"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling = pooling

        # Feature embedding
        self.in_proj = nn.Linear(input_dim, d_model)

        # Optional CLS token (only if pooling == "cls")
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # important: inputs are [B, T, D]
            norm_first=True,   # pre-norm stabilizes training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Optional final LayerNorm for extra stability
        self.norm = nn.LayerNorm(d_model) if layer_norm_final else nn.Identity()

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, F] float tensor
        returns logits: [B]
        """
        b, t, f = x.shape
        assert f == self.input_dim, f"Expected last dim {self.input_dim}, got {f}"

        h = self.in_proj(x)  # [B, T, D]

        if self.pooling == "cls":
            # prepend learnable [CLS] token
            cls = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
            h = torch.cat([cls, h], dim=1)          # [B, 1+T, D]

        h = self.pos_enc(h)                        # add PE
        h = self.encoder(h)                        # [B, T', D]
        h = self.norm(h)

        if self.pooling == "cls":
            # take the CLS position
            pooled = h[:, 0, :]                    # [B, D]
        else:
            pooled = h.mean(dim=1)                 # [B, D]

        logits = self.head(pooled).squeeze(-1)     # [B]
        return logits
