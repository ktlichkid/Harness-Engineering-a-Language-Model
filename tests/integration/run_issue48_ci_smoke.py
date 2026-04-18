"""Deterministic CI smoke training run with a bounded final-loss gate."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from small_scale_llm.training import TrainCliOverrides, prepare_training_run, run_prepared_training

MAX_FINAL_LOSS = 2.0


def _write_tinystories_fixture(dataset_path: Path) -> None:
    dataset_path.write_text(
        (
            "tiny story one here<|endoftext|>"
            "tiny story two here<|endoftext|>"
            "tiny story three here<|endoftext|>"
        ),
        encoding="utf-8",
    )


def _write_tinystories_config(config_path: Path, dataset_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "roneneldan/TinyStories",
                "cache_dir": str(dataset_path.parent),
                "splits": {
                    "train": {
                        "filename": dataset_path.name,
                        "source_url": "https://example.invalid/tinystories-smoke.txt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_train_config(
    config_path: Path,
    *,
    tinystories_config_path: Path,
    output_dir: Path,
) -> None:
    config_path.write_text(
        json.dumps(
            {
                "tinystories_config_path": str(tinystories_config_path),
                "tinystories_split": "train",
                "output_dir": str(output_dir),
                "device": "cpu",
                "download_data": False,
                "model": {
                    "hidden_size": 16,
                    "num_heads": 4,
                    "intermediate_size": 32,
                    "num_layers": 1,
                },
                "optimizer": {"learning_rate": 0.005, "weight_decay": 0.0},
                "training": {
                    "seed": 7,
                    "total_steps": 4,
                    "batch_size": 1,
                    "checkpoint_interval": 2,
                    "target_vocab_size": 48,
                    "sequence_length": 4,
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_path = root / "TinyStories-train.txt"
        tinystories_config_path = root / "tinystories.json"
        train_config_path = root / "train_tinystories_smoke.json"
        output_dir = root / "artifacts"

        _write_tinystories_fixture(dataset_path)
        _write_tinystories_config(tinystories_config_path, dataset_path)
        _write_train_config(
            train_config_path,
            tinystories_config_path=tinystories_config_path,
            output_dir=output_dir,
        )

        run = prepare_training_run(
            train_config_path,
            overrides=TrainCliOverrides(
                tinystories_path=dataset_path,
                download_data=False,
            ),
        )
        summary = run_prepared_training(run)

        if not summary.logs:
            raise AssertionError("smoke run produced no training logs")
        final_loss = float(summary.logs[-1]["loss"])
        if not math.isfinite(final_loss):
            raise AssertionError(f"smoke final loss must be finite, got {final_loss}")
        if final_loss > MAX_FINAL_LOSS:
            raise AssertionError(
                f"smoke final loss {final_loss:.6f} exceeded gate {MAX_FINAL_LOSS:.6f}"
            )

        print(
            json.dumps(
                {
                    "final_loss": final_loss,
                    "max_final_loss": MAX_FINAL_LOSS,
                    "steps": summary.completed_steps,
                    "latest_checkpoint_step": (
                        None
                        if summary.latest_checkpoint is None
                        else summary.latest_checkpoint.step
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
