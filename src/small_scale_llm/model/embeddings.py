"""Token and position embedding modules for Milestone 1."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class TokenEmbedding(Module):
    """Embedding lookup backed by a trainable token-weight table."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(vocab_size, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length).")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must use an integer dtype.")
        if token_ids.numel() == 0:
            return self.weight.new_empty((*token_ids.shape, self.embedding_dim))
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("token_ids contain values outside the embedding vocabulary.")

        flat_token_ids = token_ids.reshape(-1)
        embedded = torch.index_select(self.weight, 0, flat_token_ids)
        return embedded.reshape(*token_ids.shape, self.embedding_dim)


class PositionEmbedding(Module):
    """Trainable positional embedding table for bounded sequence lengths."""

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(max_sequence_length, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length).")

        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence_length exceeds the configured maximum.")
        if sequence_length == 0:
            return self.weight.new_empty((batch_size, 0, self.embedding_dim))

        position_ids = torch.arange(sequence_length, device=self.weight.device, dtype=torch.int64)
        position_vectors = torch.index_select(self.weight, 0, position_ids)
        return position_vectors.unsqueeze(0).expand(batch_size, -1, -1)


class TokenPositionEmbedding(Module):
    """Convenience composition boundary for token plus position embeddings."""

    def __init__(self, vocab_size: int, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = PositionEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.token_embedding(token_ids) + self.position_embedding(token_ids)
