# src/models/lstm.py
from typing import Optional
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Simple LSTM binary classifier for next-step direction.
    Expects inputs with shape: (batch, seq_len, input_dim).
    Returns logits of shape: (batch,) suitable for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        proj_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head = proj_head if proj_head is not None else nn.Linear(hidden_size * direction_factor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)           # out: (B, T, H*(1 or 2))
        last = out[:, -1, :]            # last hidden state at final timestep
        last = self.dropout(last)
        logits = self.head(last).squeeze(-1)  # (B,)
        return logits
