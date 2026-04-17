"""Loss primitives for the Milestone 1 language-model stack."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute token-level cross-entropy without forbidden functional helpers."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, sequence]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must align on batch and sequence dimensions")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must use an integer dtype")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    batch_size, sequence_length, vocab_size = logits.shape
    flat_logits = logits.reshape(batch_size * sequence_length, vocab_size)
    flat_targets = targets.reshape(batch_size * sequence_length).to(torch.int64)

    valid_mask = torch.ones_like(flat_targets, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask &= flat_targets != ignore_index

    if torch.any(flat_targets[valid_mask] < 0) or torch.any(flat_targets[valid_mask] >= vocab_size):
        raise ValueError("targets contain values outside the logits vocabulary")

    losses = logits.new_zeros(flat_targets.shape, dtype=logits.dtype)
    if torch.any(valid_mask):
        valid_logits = flat_logits[valid_mask]
        valid_targets = flat_targets[valid_mask]
        log_partition = torch.logsumexp(valid_logits, dim=-1)
        target_logits = valid_logits.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        losses[valid_mask] = log_partition - target_logits

    losses = losses.view(batch_size, sequence_length)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if not torch.any(valid_mask):
        return logits.new_tensor(0.0)
    return losses.sum() / valid_mask.sum()
