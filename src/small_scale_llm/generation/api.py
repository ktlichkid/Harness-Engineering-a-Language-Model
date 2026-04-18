"""Public trained-model generation API for Milestone 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from small_scale_llm.checkpointing import load_model_checkpoint
from small_scale_llm.model import TransformerLanguageModel
from small_scale_llm.tokenizer import BPETokenizer, load_bpe_tokenizer
from small_scale_llm.training.entrypoint import DEFAULT_SEQUENCE_LENGTH


@dataclass(frozen=True)
class StoryGenerator:
    """Minimal greedy text generator backed by a trained checkpoint."""

    model: TransformerLanguageModel
    tokenizer: BPETokenizer
    output_dir: Path
    tokenizer_path: Path
    checkpoint_path: Path
    device: torch.device
    max_sequence_length: int

    def generate(self, prompt: str, *, max_new_tokens: int = 32) -> str:
        """Generate story text by greedily extending the prompt tokens."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.model.eval()
        generated = list(self.tokenizer.encode(prompt))
        if not generated:
            raise ValueError("prompt must produce at least one token")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = generated[-self.max_sequence_length :]
                inputs = torch.tensor([window], dtype=torch.int64, device=self.device)
                logits = self.model(inputs)
                next_token = int(logits[0, -1].argmax().item())
                generated.append(next_token)

        return self.tokenizer.decode(generated)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for generation but no CUDA runtime is available.")
    return resolved


def _load_generation_metadata(
    output_dir: str | Path,
    checkpoint_path: str | Path | None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    resolved_output_dir = Path(output_dir).resolve()
    state_path = resolved_output_dir / "training_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"training state is missing at {state_path}")

    state = _read_json(state_path)
    tokenizer_path = Path(state["tokenizer_path"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer is missing at {tokenizer_path}")

    if checkpoint_path is None:
        latest_checkpoint = state.get("latest_checkpoint")
        if latest_checkpoint is None:
            raise ValueError(f"training state at {state_path} does not record a latest checkpoint")
        resolved_checkpoint_path = Path(latest_checkpoint["model_path"])
    else:
        resolved_checkpoint_path = Path(checkpoint_path)

    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"model checkpoint is missing at {resolved_checkpoint_path}")

    resolved_config_path = resolved_output_dir / "resolved_train_config.json"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"resolved training config is missing at {resolved_config_path}")

    resolved_config = _read_json(resolved_config_path)
    return resolved_output_dir, tokenizer_path, resolved_checkpoint_path, resolved_config


def load_story_generator(
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> StoryGenerator:
    """Load the public generation API from a completed training output directory."""

    resolved_output_dir, tokenizer_path, model_checkpoint_path, resolved_config = (
        _load_generation_metadata(output_dir, checkpoint_path)
    )
    config = resolved_config["config"]
    model_config = config["model"]
    training_config = config["training"]

    tokenizer = load_bpe_tokenizer(tokenizer_path)
    resolved_device = _resolve_device(device)
    max_sequence_length = int(training_config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.vocab),
        max_sequence_length=max_sequence_length,
        hidden_size=int(model_config["hidden_size"]),
        num_heads=int(model_config["num_heads"]),
        intermediate_size=int(model_config["intermediate_size"]),
        num_layers=int(model_config["num_layers"]),
    ).to(device=resolved_device)
    load_model_checkpoint(model, model_checkpoint_path)

    return StoryGenerator(
        model=model,
        tokenizer=tokenizer,
        output_dir=resolved_output_dir,
        tokenizer_path=tokenizer_path,
        checkpoint_path=model_checkpoint_path,
        device=resolved_device,
        max_sequence_length=max_sequence_length,
    )


def generate_story(
    output_dir: str | Path,
    prompt: str,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    max_new_tokens: int = 32,
) -> str:
    """Load a trained model and generate story text for one prompt."""

    generator = load_story_generator(
        output_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return generator.generate(prompt, max_new_tokens=max_new_tokens)
