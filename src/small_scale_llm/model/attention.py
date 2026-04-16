"""Attention math primitives for the Milestone 1 transformer stack."""

from __future__ import annotations

import math
from typing import Optional

import torch


def apply_linear_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a deterministic linear projection to batched sequence states."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [hidden, output]")
    if hidden_states.size(-1) != weight.size(0):
        raise ValueError("hidden state width must match the projection input width")
    if bias is not None and bias.shape != (weight.size(1),):
        raise ValueError("bias must have shape [output]")

    projected = hidden_states @ weight
    if bias is not None:
        projected = projected + bias
    return projected


def split_attention_heads(projected_states: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape projected states into a multi-head attention view."""
    if projected_states.ndim != 3:
        raise ValueError("projected_states must have shape [batch, sequence, hidden]")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if projected_states.size(-1) % num_heads != 0:
        raise ValueError("hidden width must be divisible by num_heads")

    batch_size, sequence_length, hidden_size = projected_states.shape
    head_size = hidden_size // num_heads
    return projected_states.view(batch_size, sequence_length, num_heads, head_size).permute(
        0, 2, 1, 3
    )


def merge_attention_heads(head_states: torch.Tensor) -> torch.Tensor:
    """Collapse a multi-head attention view back to sequence-major hidden states."""
    if head_states.ndim != 4:
        raise ValueError("head_states must have shape [batch, heads, sequence, head]")

    batch_size, num_heads, sequence_length, head_size = head_states.shape
    return (
        head_states.permute(0, 2, 1, 3)
        .contiguous()
        .view(batch_size, sequence_length, num_heads * head_size)
    )


def build_causal_attention_mask(
    sequence_length: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a broadcastable lower-triangular mask for causal self-attention."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
    return torch.tril(mask).view(1, 1, sequence_length, sequence_length)


def compute_attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute scaled dot-product attention scores."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, heads, sequence, head]")
    if query.shape[:-1] != key.shape[:-1]:
        raise ValueError("query and key must align on batch, heads, and sequence dimensions")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key must have the same head width")

    scale = math.sqrt(query.size(-1))
    return torch.matmul(query, key.transpose(-2, -1)) / scale


def apply_attention_mask(
    scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask disallowed attention positions with negative infinity before softmax."""
    if attention_mask is None:
        return scores
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")

    return scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked scaled dot-product attention context and weights."""
    if value.ndim != 4:
        raise ValueError("value must have shape [batch, heads, sequence, head]")
    if query.shape[:-1] != value.shape[:-1]:
        raise ValueError("query and value must align on batch, heads, and sequence dimensions")
    if query.size(-1) != value.size(-1):
        raise ValueError("query and value must have the same head width")

    masked_scores = apply_attention_mask(compute_attention_scores(query, key), attention_mask)
    attention_weights = torch.softmax(masked_scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights


def project_attention_inputs(
    hidden_states: torch.Tensor,
    *,
    num_heads: int,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    query_bias: Optional[torch.Tensor] = None,
    key_bias: Optional[torch.Tensor] = None,
    value_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project hidden states into query, key, and value head views."""
    query = split_attention_heads(
        apply_linear_projection(hidden_states, query_weight, query_bias),
        num_heads=num_heads,
    )
    key = split_attention_heads(
        apply_linear_projection(hidden_states, key_weight, key_bias),
        num_heads=num_heads,
    )
    value = split_attention_heads(
        apply_linear_projection(hidden_states, value_weight, value_bias),
        num_heads=num_heads,
    )
    return query, key, value


def project_attention_output(
    attention_context: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Merge attention heads and apply the output projection."""
    return apply_linear_projection(
        merge_attention_heads(attention_context),
        output_weight,
        output_bias,
    )
