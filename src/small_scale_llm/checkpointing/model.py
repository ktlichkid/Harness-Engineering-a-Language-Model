"""Deterministic model checkpoint save/load helpers for Milestone 1."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import Module


def save_model_checkpoint(model: Module, output_path: str | Path) -> Path:
    """Persist a model state dict to a deterministic checkpoint file."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state_dict, destination)
    return destination


def load_model_checkpoint(model: Module, input_path: str | Path) -> list[str]:
    """Load a checkpoint into the provided model and return loaded parameter keys."""
    checkpoint = torch.load(Path(input_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a state-dict mapping")
    if not all(isinstance(name, str) for name in checkpoint):
        raise ValueError("checkpoint keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError("checkpoint values must be tensors")

    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("checkpoint payload does not match the model state")
    return sorted(checkpoint.keys())
