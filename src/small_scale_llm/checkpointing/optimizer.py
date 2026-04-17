"""Optimizer checkpoint save/load helpers for Milestone 1."""

from __future__ import annotations

from pathlib import Path

import torch

from small_scale_llm.optim import AdamW


def save_optimizer_checkpoint(optimizer: AdamW, output_path: str | Path) -> Path:
    """Persist optimizer state deterministically with torch.save."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), destination)
    return destination


def load_optimizer_checkpoint(
    optimizer: AdamW,
    input_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load optimizer state into an existing optimizer instance."""
    checkpoint = torch.load(Path(input_path), map_location=map_location)
    optimizer.load_state_dict(checkpoint)
    return checkpoint
