"""Public TinyStories training entry-point contract for Milestone 1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from small_scale_llm.checkpointing import (
    load_model_checkpoint,
    load_optimizer_checkpoint,
    save_model_checkpoint,
    save_optimizer_checkpoint,
)
from small_scale_llm.data import load_tinystories_config, materialize_tinystories_split
from small_scale_llm.data.tinystories import iter_tinystories_records
from small_scale_llm.model import TransformerLanguageModel
from small_scale_llm.optim import AdamW
from small_scale_llm.tokenizer import (
    BPETokenizer,
    load_bpe_tokenizer,
    train_bpe_from_tinystories,
    write_bpe_artifact,
)
from small_scale_llm.training.step import run_training_step

DEFAULT_TRAIN_CONFIG_PATH = Path("configs/milestone1/train_tinystories.json")
DEFAULT_SEQUENCE_LENGTH = 32


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_heads: int
    intermediate_size: int
    num_layers: int

    def as_dict(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
        }


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float

    def as_dict(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class TrainingLoopConfig:
    seed: int
    total_steps: int
    batch_size: int
    checkpoint_interval: int
    target_vocab_size: int
    sequence_length: int

    def as_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "total_steps": self.total_steps,
            "batch_size": self.batch_size,
            "checkpoint_interval": self.checkpoint_interval,
            "target_vocab_size": self.target_vocab_size,
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True)
class TinyStoriesTrainingConfig:
    tinystories_config_path: Path
    tinystories_split: str
    output_dir: Path
    device: str
    download_data: bool
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingLoopConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "tinystories_config_path": str(self.tinystories_config_path),
            "tinystories_split": self.tinystories_split,
            "output_dir": str(self.output_dir),
            "device": self.device,
            "download_data": self.download_data,
            "model": self.model.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "training": self.training.as_dict(),
        }


@dataclass(frozen=True)
class ResumeCheckpoint:
    step: int
    model_path: Path
    optimizer_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "model_path": str(self.model_path),
            "optimizer_path": str(self.optimizer_path),
        }


@dataclass(frozen=True)
class TrainCliOverrides:
    output_dir: Path | None = None
    tinystories_path: Path | None = None
    device: str | None = None
    download_data: bool | None = None
    resume: bool = False


@dataclass(frozen=True)
class PreparedTrainingRun:
    config_path: Path
    config: TinyStoriesTrainingConfig
    dataset_path: Path
    output_dir: Path
    checkpoints_dir: Path
    resolved_config_path: Path
    state_path: Path
    resume_checkpoint: ResumeCheckpoint | None

    def as_dict(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "config": self.config.as_dict(),
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "resolved_config_path": str(self.resolved_config_path),
            "state_path": str(self.state_path),
            "resume_checkpoint": (
                None if self.resume_checkpoint is None else self.resume_checkpoint.as_dict()
            ),
        }


@dataclass(frozen=True)
class TrainingExecutionSummary:
    config_path: Path
    output_dir: Path
    dataset_path: Path
    tokenizer_path: Path
    checkpoints_dir: Path
    start_step: int
    end_step: int
    completed_steps: int
    device: str
    latest_checkpoint: ResumeCheckpoint | None
    logs: list[dict[str, Any]]

    def as_dict(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "dataset_path": str(self.dataset_path),
            "tokenizer_path": str(self.tokenizer_path),
            "checkpoints_dir": str(self.checkpoints_dir),
            "start_step": self.start_step,
            "end_step": self.end_step,
            "completed_steps": self.completed_steps,
            "device": self.device,
            "latest_checkpoint": (
                None if self.latest_checkpoint is None else self.latest_checkpoint.as_dict()
            ),
            "logs": self.logs,
        }


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_training_config(config_path: str | Path) -> TinyStoriesTrainingConfig:
    """Load the checked-in public training config contract."""

    config_path = Path(config_path)
    config_data = json.loads(config_path.read_text(encoding="utf-8-sig"))

    model_config = ModelConfig(
        hidden_size=_require_positive_int(config_data["model"]["hidden_size"], "model.hidden_size"),
        num_heads=_require_positive_int(config_data["model"]["num_heads"], "model.num_heads"),
        intermediate_size=_require_positive_int(
            config_data["model"]["intermediate_size"],
            "model.intermediate_size",
        ),
        num_layers=_require_positive_int(config_data["model"]["num_layers"], "model.num_layers"),
    )
    optimizer_config = OptimizerConfig(
        learning_rate=float(config_data["optimizer"]["learning_rate"]),
        weight_decay=float(config_data["optimizer"]["weight_decay"]),
    )
    training_config = TrainingLoopConfig(
        seed=_require_positive_int(config_data["training"]["seed"], "training.seed"),
        total_steps=_require_positive_int(
            config_data["training"]["total_steps"],
            "training.total_steps",
        ),
        batch_size=_require_positive_int(
            config_data["training"]["batch_size"],
            "training.batch_size",
        ),
        checkpoint_interval=_require_positive_int(
            config_data["training"]["checkpoint_interval"],
            "training.checkpoint_interval",
        ),
        target_vocab_size=_require_positive_int(
            config_data["training"]["target_vocab_size"],
            "training.target_vocab_size",
        ),
        sequence_length=_require_positive_int(
            config_data["training"].get("sequence_length", DEFAULT_SEQUENCE_LENGTH),
            "training.sequence_length",
        ),
    )

    device = str(config_data["device"])
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")

    return TinyStoriesTrainingConfig(
        tinystories_config_path=Path(config_data["tinystories_config_path"]),
        tinystories_split=str(config_data["tinystories_split"]),
        output_dir=Path(config_data["output_dir"]),
        device=device,
        download_data=bool(config_data["download_data"]),
        model=model_config,
        optimizer=optimizer_config,
        training=training_config,
    )


def _apply_overrides(
    config: TinyStoriesTrainingConfig,
    overrides: TrainCliOverrides,
) -> TinyStoriesTrainingConfig:
    return TinyStoriesTrainingConfig(
        tinystories_config_path=config.tinystories_config_path,
        tinystories_split=config.tinystories_split,
        output_dir=config.output_dir if overrides.output_dir is None else overrides.output_dir,
        device=config.device if overrides.device is None else overrides.device,
        download_data=(
            config.download_data if overrides.download_data is None else overrides.download_data
        ),
        model=config.model,
        optimizer=config.optimizer,
        training=config.training,
    )


def _load_resume_checkpoint(state_path: Path) -> ResumeCheckpoint:
    if not state_path.exists():
        raise FileNotFoundError(f"resume state is missing at {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_data = state.get("latest_checkpoint")
    if checkpoint_data is None:
        raise ValueError(f"resume state at {state_path} does not record a latest checkpoint")

    checkpoint = ResumeCheckpoint(
        step=_require_positive_int(checkpoint_data["step"], "latest_checkpoint.step"),
        model_path=Path(checkpoint_data["model_path"]),
        optimizer_path=Path(checkpoint_data["optimizer_path"]),
    )
    if not checkpoint.model_path.exists():
        raise FileNotFoundError(f"resume model checkpoint is missing at {checkpoint.model_path}")
    if not checkpoint.optimizer_path.exists():
        raise FileNotFoundError(
            f"resume optimizer checkpoint is missing at {checkpoint.optimizer_path}"
        )
    return checkpoint


def _build_initial_state(run: PreparedTrainingRun) -> dict[str, object]:
    return {
        "config_path": str(run.config_path),
        "dataset_path": str(run.dataset_path),
        "output_dir": str(run.output_dir),
        "checkpoints_dir": str(run.checkpoints_dir),
        "latest_checkpoint": (
            None if run.resume_checkpoint is None else run.resume_checkpoint.as_dict()
        ),
    }


def prepare_training_run(
    config_path: str | Path,
    overrides: TrainCliOverrides | None = None,
) -> PreparedTrainingRun:
    """Resolve the public training contract into concrete local paths."""

    overrides = TrainCliOverrides() if overrides is None else overrides
    loaded_config = load_training_config(config_path)
    config = _apply_overrides(loaded_config, overrides)

    tinystories_config = load_tinystories_config(
        config.tinystories_config_path,
        config.tinystories_split,
    )
    if overrides.tinystories_path is None:
        dataset_path = materialize_tinystories_split(
            tinystories_config,
            download=config.download_data,
        )
    else:
        dataset_path = Path(overrides.tinystories_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"TinyStories dataset path does not exist: {dataset_path}")

    output_dir = config.output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    resolved_config_path = output_dir / "resolved_train_config.json"
    state_path = output_dir / "training_state.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = _load_resume_checkpoint(state_path) if overrides.resume else None

    prepared_run = PreparedTrainingRun(
        config_path=Path(config_path).resolve(),
        config=TinyStoriesTrainingConfig(
            tinystories_config_path=config.tinystories_config_path.resolve(),
            tinystories_split=config.tinystories_split,
            output_dir=output_dir,
            device=config.device,
            download_data=config.download_data,
            model=config.model,
            optimizer=config.optimizer,
            training=config.training,
        ),
        dataset_path=Path(dataset_path).resolve(),
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        resolved_config_path=resolved_config_path,
        state_path=state_path,
        resume_checkpoint=resume_checkpoint,
    )

    _write_json(resolved_config_path, prepared_run.as_dict())
    if not state_path.exists():
        _write_json(state_path, _build_initial_state(prepared_run))
    return prepared_run


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA runtime is available.")
    return torch.device(device_name)


def _prepare_tokenizer(
    run: PreparedTrainingRun,
    *,
    resume: bool,
) -> tuple[BPETokenizer, Path]:
    tokenizer_path = run.output_dir / "tokenizer.json"
    artifact_path = run.output_dir / "tokenizer_artifact.json"

    if tokenizer_path.exists():
        return load_bpe_tokenizer(tokenizer_path), tokenizer_path
    if resume:
        raise FileNotFoundError(f"resume tokenizer is missing at {tokenizer_path}")

    artifact = train_bpe_from_tinystories(
        run.dataset_path,
        split=run.config.tinystories_split,
        target_vocab_size=run.config.training.target_vocab_size,
    )
    write_bpe_artifact(artifact, artifact_path)
    tokenizer = BPETokenizer.from_artifact(artifact)
    tokenizer.save(tokenizer_path)
    return tokenizer, tokenizer_path


def _collect_training_tokens(run: PreparedTrainingRun, tokenizer: BPETokenizer) -> list[int]:
    token_ids: list[int] = []
    for record in iter_tinystories_records(run.dataset_path, run.config.tinystories_split):
        encoded = tokenizer.encode(record.text)
        if encoded:
            token_ids.extend(encoded)
    if not token_ids:
        raise ValueError("TinyStories dataset produced no token ids for training.")
    return token_ids


def _build_token_windows(
    token_ids: list[int],
    *,
    sequence_length: int,
) -> list[list[int]]:
    required_length = sequence_length + 1
    if len(token_ids) < required_length:
        raise ValueError(
            "TinyStories token stream is too short for training.sequence_length "
            f"(need at least {required_length} tokens, found {len(token_ids)})."
        )

    windows: list[list[int]] = []
    stride = sequence_length
    final_start = len(token_ids) - required_length
    for start in range(0, final_start + 1, stride):
        windows.append(token_ids[start : start + required_length])

    if not windows:
        windows.append(token_ids[:required_length])
    return windows


def _build_batch_tensor(
    windows: list[list[int]],
    *,
    batch_size: int,
    step_index: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[int]] = []
    offset = step_index * batch_size
    for index in range(batch_size):
        rows.append(windows[(offset + index) % len(windows)])
    return torch.tensor(rows, dtype=torch.int64, device=device)


def _save_training_state(
    run: PreparedTrainingRun,
    *,
    tokenizer_path: Path,
    latest_checkpoint: ResumeCheckpoint | None,
) -> None:
    _write_json(
        run.state_path,
        {
            "config_path": str(run.config_path),
            "dataset_path": str(run.dataset_path),
            "output_dir": str(run.output_dir),
            "checkpoints_dir": str(run.checkpoints_dir),
            "tokenizer_path": str(tokenizer_path),
            "latest_checkpoint": (
                None if latest_checkpoint is None else latest_checkpoint.as_dict()
            ),
        },
    )


def run_prepared_training(run: PreparedTrainingRun) -> TrainingExecutionSummary:
    """Run the end-to-end public TinyStories training path for the prepared run."""

    torch.manual_seed(run.config.training.seed)
    device = _resolve_device(run.config.device)
    tokenizer, tokenizer_path = _prepare_tokenizer(run, resume=run.resume_checkpoint is not None)
    token_stream = _collect_training_tokens(run, tokenizer)
    windows = _build_token_windows(
        token_stream,
        sequence_length=run.config.training.sequence_length,
    )

    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.vocab),
        max_sequence_length=run.config.training.sequence_length,
        hidden_size=run.config.model.hidden_size,
        num_heads=run.config.model.num_heads,
        intermediate_size=run.config.model.intermediate_size,
        num_layers=run.config.model.num_layers,
    ).to(device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=run.config.optimizer.learning_rate,
        weight_decay=run.config.optimizer.weight_decay,
    )

    start_step = 0
    if run.resume_checkpoint is not None:
        load_model_checkpoint(model, run.resume_checkpoint.model_path)
        load_optimizer_checkpoint(optimizer, run.resume_checkpoint.optimizer_path)
        start_step = run.resume_checkpoint.step

    latest_checkpoint = run.resume_checkpoint
    logs: list[dict[str, Any]] = []
    total_steps = run.config.training.total_steps

    for step_index in range(start_step, total_steps):
        token_batch = _build_batch_tensor(
            windows,
            batch_size=run.config.training.batch_size,
            step_index=step_index,
            device=device,
        )
        logs.append(
            run_training_step(
                model,
                optimizer,
                token_batch,
                step_index=step_index,
            )
        )

        completed_step = step_index + 1
        should_checkpoint = (
            completed_step % run.config.training.checkpoint_interval == 0
            or completed_step == total_steps
        )
        if should_checkpoint:
            model_path = run.checkpoints_dir / f"model-step-{completed_step}.pt"
            optimizer_path = run.checkpoints_dir / f"optimizer-step-{completed_step}.pt"
            save_model_checkpoint(model, model_path)
            save_optimizer_checkpoint(optimizer, optimizer_path)
            latest_checkpoint = ResumeCheckpoint(
                step=completed_step,
                model_path=model_path,
                optimizer_path=optimizer_path,
            )
            _save_training_state(
                run,
                tokenizer_path=tokenizer_path,
                latest_checkpoint=latest_checkpoint,
            )

    if latest_checkpoint is None:
        _save_training_state(
            run,
            tokenizer_path=tokenizer_path,
            latest_checkpoint=None,
        )

    end_step = start_step if not logs else logs[-1]["step"] + 1
    return TrainingExecutionSummary(
        config_path=run.config_path,
        output_dir=run.output_dir,
        dataset_path=run.dataset_path,
        tokenizer_path=tokenizer_path,
        checkpoints_dir=run.checkpoints_dir,
        start_step=start_step,
        end_step=end_step,
        completed_steps=len(logs),
        device=str(device),
        latest_checkpoint=latest_checkpoint,
        logs=logs,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the narrow public CLI contract for the training entry point."""

    parser = argparse.ArgumentParser(
        description="Prepare the public TinyStories training run contract.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAIN_CONFIG_PATH),
        help="Path to the TinyStories training config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for the training output directory.",
    )
    parser.add_argument(
        "--tinystories-path",
        help="Optional override for the local TinyStories split file.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Optional override for the configured device.",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download-data",
        action="store_true",
        dest="download_data",
        default=None,
        help="Download the configured TinyStories split if it is missing.",
    )
    download_group.add_argument(
        "--no-download-data",
        action="store_false",
        dest="download_data",
        help="Require the configured TinyStories split to exist locally.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the latest checkpoint recorded in output_dir/training_state.json.",
    )
    return parser


def _build_cli_overrides(arguments: argparse.Namespace) -> TrainCliOverrides:
    return TrainCliOverrides(
        output_dir=None if arguments.output_dir is None else Path(arguments.output_dir),
        tinystories_path=(
            None if arguments.tinystories_path is None else Path(arguments.tinystories_path)
        ),
        device=arguments.device,
        download_data=arguments.download_data,
        resume=arguments.resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public TinyStories training entry point end to end."""

    arguments = build_argument_parser().parse_args(argv)
    prepared_run = prepare_training_run(
        arguments.config,
        overrides=_build_cli_overrides(arguments),
    )
    summary = run_prepared_training(prepared_run)
    print(json.dumps(summary.as_dict(), indent=2))
    print("Completed TinyStories end-to-end training run.")
    return 0
