"""Feed-forward primitives for the Milestone 1 transformer stack."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .attention import apply_linear_projection


def gelu(hidden_states: Tensor) -> Tensor:
    """Apply the exact GELU activation without torch.nn.functional."""
    return 0.5 * hidden_states * (1.0 + torch.erf(hidden_states / math.sqrt(2.0)))


class FeedForwardNetwork(Module):
    """Two-layer transformer feed-forward path with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.input_weight = Parameter(torch.empty(hidden_size, intermediate_size))
        self.input_bias = Parameter(torch.zeros(intermediate_size, dtype=torch.float32))
        self.output_weight = Parameter(torch.empty(intermediate_size, hidden_size))
        self.output_bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        input_bound = 1.0 / math.sqrt(self.hidden_size)
        output_bound = 1.0 / math.sqrt(self.intermediate_size)
        with torch.no_grad():
            self.input_weight.uniform_(-input_bound, input_bound)
            self.output_weight.uniform_(-output_bound, output_bound)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_projection = apply_linear_projection(
            hidden_states,
            self.input_weight,
            self.input_bias,
        )
        activated = gelu(hidden_projection)
        return apply_linear_projection(
            activated,
            self.output_weight,
            self.output_bias,
        )
