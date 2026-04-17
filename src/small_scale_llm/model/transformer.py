"""Transformer model assembly for Milestone 1."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from .attention import (
    build_causal_attention_mask,
    project_attention_inputs,
    project_attention_output,
    scaled_dot_product_attention,
)
from .embeddings import TokenPositionEmbedding
from .feedforward import FeedForwardNetwork
from .normalization import LayerNorm


class CausalSelfAttention(Module):
    """Assembled causal self-attention from the merged attention primitives."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query_weight = Parameter(torch.empty(hidden_size, hidden_size))
        self.query_bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.key_weight = Parameter(torch.empty(hidden_size, hidden_size))
        self.key_bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.value_weight = Parameter(torch.empty(hidden_size, hidden_size))
        self.value_bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.output_weight = Parameter(torch.empty(hidden_size, hidden_size))
        self.output_bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            self.query_weight.uniform_(-bound, bound)
            self.key_weight.uniform_(-bound, bound)
            self.value_weight.uniform_(-bound, bound)
            self.output_weight.uniform_(-bound, bound)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError("hidden state width must match hidden_size")

        query, key, value = project_attention_inputs(
            hidden_states,
            num_heads=self.num_heads,
            query_weight=self.query_weight,
            key_weight=self.key_weight,
            value_weight=self.value_weight,
            query_bias=self.query_bias,
            key_bias=self.key_bias,
            value_bias=self.value_bias,
        )
        sequence_length = hidden_states.size(1)
        attention_mask = build_causal_attention_mask(
            sequence_length=sequence_length,
            device=hidden_states.device,
        )
        context, _ = scaled_dot_product_attention(query, key, value, attention_mask)
        return project_attention_output(context, self.output_weight, self.output_bias)


class TransformerBlock(Module):
    """Single transformer block assembled from attention, normalization, and FFN primitives."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(hidden_size=hidden_size)
        self.attention = CausalSelfAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.feedforward_norm = LayerNorm(hidden_size=hidden_size)
        self.feedforward = FeedForwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_input = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.attention(attention_input)
        feedforward_input = self.feedforward_norm(hidden_states)
        return hidden_states + self.feedforward(feedforward_input)


class TransformerLanguageModel(Module):
    """Language-model stack assembled from the merged Milestone 1 primitives."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dim=hidden_size,
        )
        self.blocks = ModuleList(
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_layers)
        )
        self.final_norm = LayerNorm(hidden_size=hidden_size)
        self.output_weight = Parameter(torch.empty(hidden_size, vocab_size))
        self.output_bias = Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            self.output_weight.uniform_(-bound, bound)

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden_states = self.embedding(token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        normalized = self.final_norm(hidden_states)
        return normalized @ self.output_weight + self.output_bias
