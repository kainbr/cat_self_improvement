import math

import torch
import torch.nn as nn
from torch import Tensor


class AbsolutePositionEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:

        assert x.dim() == 4
        B, T, m, d = x.shape
        assert positions.shape == (B, T, m)
        assert d == self.d_model == self.pe.shape[-1]
        assert positions.max() < self.max_len

        if torch.any(positions < 0):
            raise NotImplementedError  # handle, e.g. by setting explicitly to zero

        return x + self.pe[positions]
