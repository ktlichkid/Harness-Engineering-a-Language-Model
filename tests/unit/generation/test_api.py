import json
import tempfile
import unittest
from pathlib import Path

from small_scale_llm.generation import generate_story, load_story_generator
from small_scale_llm.training import (
    TrainCliOverrides,
    prepare_training_run,
    run_prepared_training,
)


class GenerationApiTests(unittest.TestCase):
    def test_load_story_generator_uses_latest_checkpoint_and_generates_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = self._run_training_fixture(Path(temp_dir), total_steps=3)

            generator = load_story_generator(output_dir, device="cpu")
            generated = generator.generate("hello world", max_new_tokens=4)

            self.assertEqual(generator.device.type, "cpu")
            self.assertTrue(generator.checkpoint_path.exists())
            self.assertEqual(generator.checkpoint_path.name, "model-step-3.pt")
            self.assertTrue(generated.startswith("hello world"))
            self.assertGreaterEqual(len(generated.split()), 2)

    def test_generate_story_accepts_explicit_checkpoint_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = self._run_training_fixture(Path(temp_dir), total_steps=3)
            checkpoint_path = output_dir / "checkpoints" / "model-step-1.pt"

            generator = load_story_generator(
                output_dir,
                checkpoint_path=checkpoint_path,
                device="cpu",
            )
            generated = generate_story(
                output_dir,
                "alpha beta",
                checkpoint_path=checkpoint_path,
                device="cpu",
                max_new_tokens=3,
            )

            self.assertEqual(generator.checkpoint_path, checkpoint_path)
            self.assertTrue(generated.startswith("alpha beta"))

    def _run_training_fixture(self, root: Path, *, total_steps: int) -> Path:
        dataset_path = root / "TinyStories-train.txt"
        dataset_path.write_text(
            "hello world hello world<|endoftext|>"
            "alpha beta alpha beta<|endoftext|>"
            "gentle stories stay short<|endoftext|>",
            encoding="utf-8",
        )
        tinystories_config_path = self._write_tinystories_config(root, dataset_path)
        output_dir = root / "artifacts"
        training_config_path = self._write_training_config(
            root,
            tinystories_config_path,
            output_dir,
            total_steps=total_steps,
        )
        run = prepare_training_run(
            training_config_path,
            overrides=TrainCliOverrides(
                tinystories_path=dataset_path,
                download_data=False,
            ),
        )

        run_prepared_training(run)
        return output_dir

    def _write_tinystories_config(self, root: Path, dataset_path: Path) -> Path:
        config_path = root / "tinystories.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_id": "roneneldan/TinyStories",
                    "cache_dir": str(dataset_path.parent),
                    "splits": {
                        "train": {
                            "filename": dataset_path.name,
                            "source_url": "https://example.invalid/tinystories-train.txt",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def _write_training_config(
        self,
        root: Path,
        tinystories_config_path: Path,
        output_dir: Path,
        *,
        total_steps: int,
    ) -> Path:
        config_path = root / "train_tinystories.json"
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
                    "optimizer": {
                        "learning_rate": 0.005,
                        "weight_decay": 0.0,
                    },
                    "training": {
                        "seed": 7,
                        "total_steps": total_steps,
                        "batch_size": 1,
                        "checkpoint_interval": 1,
                        "target_vocab_size": 48,
                        "sequence_length": 4,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
