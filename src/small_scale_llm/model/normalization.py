"""Normalization primitives for the Milestone 1 transformer stack."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class LayerNorm(Module):
    """Manual layer normalization over the final hidden dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError("hidden state width must match hidden_size")

        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias
