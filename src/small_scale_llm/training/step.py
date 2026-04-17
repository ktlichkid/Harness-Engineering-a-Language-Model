"""Training-step orchestration for the Milestone 1 language-model stack."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import Tensor

from small_scale_llm.model import TransformerLanguageModel, cross_entropy_loss
from small_scale_llm.optim import AdamW


def prepare_language_model_batch(token_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Shift a token batch into autoregressive inputs and targets."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must use an integer dtype")
    if token_ids.size(1) < 2:
        raise ValueError("token_ids must include at least two positions for next-token training")

    return token_ids[:, :-1], token_ids[:, 1:]


def compute_gradient_norm(model: TransformerLanguageModel) -> float:
    """Return the global L2 norm of currently populated gradients."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.pow(2).sum().item())
    return total**0.5


def run_training_step(
    model: TransformerLanguageModel,
    optimizer: AdamW,
    token_ids: Tensor,
    *,
    ignore_index: Optional[int] = None,
    step_index: int = 0,
) -> dict[str, Any]:
    """Run one forward/backward/update pass and return minimal debug logs."""
    inputs, targets = prepare_language_model_batch(token_ids)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets, ignore_index=ignore_index, reduction="mean")
    loss.backward()
    grad_norm = compute_gradient_norm(model)
    optimizer.step()

    valid_targets = targets
    if ignore_index is not None:
        valid_targets = targets[targets != ignore_index]

    return {
        "step": step_index,
        "loss": float(loss.detach().item()),
        "tokens": int(valid_targets.numel()),
        "grad_norm": grad_norm,
        "mean_logit": float(logits.detach().mean().item()),
    }


def run_training_loop(
    model: TransformerLanguageModel,
    optimizer: AdamW,
    batches: Iterable[Tensor],
    *,
    ignore_index: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Run a short sequence of training steps and collect per-step debug logs."""
    logs: list[dict[str, Any]] = []
    for step_index, token_ids in enumerate(batches):
        logs.append(
            run_training_step(
                model,
                optimizer,
                token_ids,
                ignore_index=ignore_index,
                step_index=step_index,
            )
        )
    return logs
