"""Minimal single-GPU Milestone 1 integration run for issue #18."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from small_scale_llm.checkpointing import (
    load_model_checkpoint,
    load_optimizer_checkpoint,
    save_model_checkpoint,
    save_optimizer_checkpoint,
)
from small_scale_llm.model import TransformerLanguageModel
from small_scale_llm.optim import AdamW
from small_scale_llm.tokenizer import BPETokenizer, train_bpe_from_texts
from small_scale_llm.training import run_training_step


@dataclass(frozen=True)
class Issue18RunConfig:
    seed: int = 18
    target_vocab_size: int = 96
    hidden_size: int = 64
    num_heads: int = 4
    intermediate_size: int = 128
    num_layers: int = 2
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    total_steps: int = 160
    resume_step: int = 80
    max_new_tokens: int = 16
    prompt: str = "once upon a time there was"


TRAINING_TEXTS = [
    "once upon a time there was a brave cat who loved warm soup",
    "once upon a time there was a kind fox who loved bright stars",
    "the brave cat found a tiny lantern and smiled at the moon",
    "the kind fox shared soft bread and told a happy story",
    "the brave cat and the kind fox walked home through gentle rain",
    "they sang a quiet song and everyone slept in a warm house",
]


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for issue #18, but torch.cuda.is_available() is False."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for issue #18, but torch.cuda.device_count() is 0.")
    return torch.device("cuda")


def build_tokenizer_and_batches(
    config: Issue18RunConfig,
    device: torch.device,
) -> tuple[BPETokenizer, list[torch.Tensor], int]:
    artifact = train_bpe_from_texts(TRAINING_TEXTS, target_vocab_size=config.target_vocab_size)
    tokenizer = BPETokenizer.from_artifact(artifact)
    encoded = [tokenizer.encode(text) for text in TRAINING_TEXTS]
    max_sequence_length = max(len(token_ids) for token_ids in encoded) - 1
    if max_sequence_length <= 0:
        raise ValueError("training texts must produce at least two tokens per sequence")

    batches = [
        torch.tensor([token_ids], dtype=torch.int64, device=device)
        for token_ids in encoded
    ]
    return tokenizer, batches, max_sequence_length


def build_model(
    config: Issue18RunConfig,
    *,
    vocab_size: int,
    max_sequence_length: int,
    device: torch.device,
) -> TransformerLanguageModel:
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        num_layers=config.num_layers,
    )
    return model.to(device)


def build_optimizer(model: TransformerLanguageModel, config: Issue18RunConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def run_steps(
    model: TransformerLanguageModel,
    optimizer: AdamW,
    batches: list[torch.Tensor],
    *,
    start_step: int,
    total_steps: int,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for step_index in range(start_step, total_steps):
        batch = batches[step_index % len(batches)]
        logs.append(run_training_step(model, optimizer, batch, step_index=step_index))
    return logs


def max_state_difference(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    max_difference = 0.0
    for name in left:
        difference = float((left[name] - right[name]).abs().max().item())
        if difference > max_difference:
            max_difference = difference
    return max_difference


def generate_story(
    model: TransformerLanguageModel,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_sequence_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = generated[-max_sequence_length:]
            inputs = torch.tensor([window], dtype=torch.int64, device=device)
            logits = model(inputs)
            next_token = int(logits[0, -1].argmax().item())
            generated.append(next_token)

    return tokenizer.decode(generated)


def run_issue18_single_gpu(output_dir: str | Path) -> dict[str, Any]:
    config = Issue18RunConfig()
    configure_determinism(config.seed)
    device = build_device()
    tokenizer, batches, max_sequence_length = build_tokenizer_and_batches(config, device)

    reference_model = build_model(
        config,
        vocab_size=len(tokenizer.vocab),
        max_sequence_length=max_sequence_length,
        device=device,
    )
    resumed_source_model = copy.deepcopy(reference_model)
    reference_optimizer = build_optimizer(reference_model, config)
    resumed_source_optimizer = build_optimizer(resumed_source_model, config)

    reference_logs = run_steps(
        reference_model,
        reference_optimizer,
        batches,
        start_step=0,
        total_steps=config.total_steps,
    )
    pre_resume_logs = run_steps(
        resumed_source_model,
        resumed_source_optimizer,
        batches,
        start_step=0,
        total_steps=config.resume_step,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = save_model_checkpoint(
        resumed_source_model,
        output_path / "model-resume-checkpoint.pt",
    )
    optimizer_checkpoint_path = save_optimizer_checkpoint(
        resumed_source_optimizer,
        output_path / "optimizer-resume-checkpoint.pt",
    )

    resumed_model = build_model(
        config,
        vocab_size=len(tokenizer.vocab),
        max_sequence_length=max_sequence_length,
        device=device,
    )
    resumed_optimizer = build_optimizer(resumed_model, config)
    loaded_model_keys = load_model_checkpoint(resumed_model, model_checkpoint_path)
    load_optimizer_checkpoint(resumed_optimizer, optimizer_checkpoint_path, map_location=device)

    resumed_logs = pre_resume_logs + run_steps(
        resumed_model,
        resumed_optimizer,
        batches,
        start_step=config.resume_step,
        total_steps=config.total_steps,
    )

    reference_state = {
        name: tensor.detach().cpu()
        for name, tensor in reference_model.state_dict().items()
    }
    resumed_state = {
        name: tensor.detach().cpu()
        for name, tensor in resumed_model.state_dict().items()
    }

    sample_story = generate_story(
        resumed_model,
        tokenizer,
        config.prompt,
        max_sequence_length=max_sequence_length,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    summary = {
        "config": asdict(config),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "loaded_model_key_count": len(loaded_model_keys),
        "resume_validation": {
            "model_max_abs_diff": max_state_difference(reference_state, resumed_state),
            "reference_final_loss": reference_logs[-1]["loss"],
            "resumed_final_loss": resumed_logs[-1]["loss"],
        },
        "generated_story": sample_story,
    }

    (output_path / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_path / "training_log.json").write_text(
        json.dumps({"reference": reference_logs, "resumed": resumed_logs}, indent=2),
        encoding="utf-8",
    )
    (output_path / "generated_story.txt").write_text(sample_story + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_issue18_single_gpu(Path("artifacts") / "issue18")
