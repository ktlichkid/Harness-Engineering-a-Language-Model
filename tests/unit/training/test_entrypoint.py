import json
import tempfile
import unittest
from pathlib import Path

from small_scale_llm.training import TrainCliOverrides, load_training_config, prepare_training_run


class TrainingEntrypointConfigTests(unittest.TestCase):
    def test_load_training_config_uses_checked_in_contract(self) -> None:
        config = load_training_config("configs/milestone1/train_tinystories.json")

        self.assertEqual(config.tinystories_split, "train")
        self.assertEqual(config.output_dir, Path("artifacts/tinystories-train"))
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.download_data)
        self.assertEqual(config.model.hidden_size, 64)
        self.assertEqual(config.optimizer.learning_rate, 0.01)
        self.assertEqual(config.training.checkpoint_interval, 80)

    def test_load_training_config_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train_tinystories.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tinystories_config_path": "configs/milestone1/tinystories.json",
                        "tinystories_split": "train",
                        "output_dir": "artifacts/tinystories-train",
                        "device": "cpu",
                        "download_data": False,
                        "model": {
                            "hidden_size": 32,
                            "num_heads": 4,
                            "intermediate_size": 64,
                            "num_layers": 2,
                        },
                        "optimizer": {
                            "learning_rate": 0.01,
                            "weight_decay": 0.0,
                        },
                        "training": {
                            "seed": 45,
                            "total_steps": 10,
                            "batch_size": 1,
                            "checkpoint_interval": 5,
                            "target_vocab_size": 64,
                        },
                    }
                ),
                encoding="utf-8-sig",
            )

            config = load_training_config(config_path)

            self.assertEqual(config.device, "cpu")
            self.assertFalse(config.download_data)


class TrainingEntrypointPreparationTests(unittest.TestCase):
    def test_prepare_training_run_creates_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "TinyStories-train.txt"
            dataset_path.write_text("a tiny story<|endoftext|>", encoding="utf-8")

            tinystories_config_path = self._write_tinystories_config(root, dataset_path)
            output_dir = root / "artifacts"
            training_config_path = self._write_training_config(
                root,
                tinystories_config_path,
                output_dir,
            )

            prepared_run = prepare_training_run(
                training_config_path,
                overrides=TrainCliOverrides(download_data=False),
            )

            self.assertEqual(prepared_run.dataset_path, dataset_path.resolve())
            self.assertEqual(prepared_run.output_dir, output_dir.resolve())
            self.assertTrue(prepared_run.checkpoints_dir.is_dir())
            self.assertTrue(prepared_run.resolved_config_path.exists())
            self.assertTrue(prepared_run.state_path.exists())
            self.assertIsNone(prepared_run.resume_checkpoint)

            state = json.loads(prepared_run.state_path.read_text(encoding="utf-8"))
            self.assertIsNone(state["latest_checkpoint"])

    def test_prepare_training_run_loads_latest_checkpoint_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "TinyStories-train.txt"
            dataset_path.write_text("a tiny story<|endoftext|>", encoding="utf-8")

            tinystories_config_path = self._write_tinystories_config(root, dataset_path)
            output_dir = root / "artifacts"
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)
            model_path = checkpoints_dir / "model-step-4.pt"
            optimizer_path = checkpoints_dir / "optimizer-step-4.pt"
            model_path.write_bytes(b"model")
            optimizer_path.write_bytes(b"optimizer")
            state_path = output_dir / "training_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "config_path": "unused",
                        "dataset_path": str(dataset_path),
                        "output_dir": str(output_dir),
                        "checkpoints_dir": str(checkpoints_dir),
                        "latest_checkpoint": {
                            "step": 4,
                            "model_path": str(model_path),
                            "optimizer_path": str(optimizer_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            training_config_path = self._write_training_config(
                root,
                tinystories_config_path,
                output_dir,
            )

            prepared_run = prepare_training_run(
                training_config_path,
                overrides=TrainCliOverrides(download_data=False, resume=True),
            )

            self.assertIsNotNone(prepared_run.resume_checkpoint)
            assert prepared_run.resume_checkpoint is not None
            self.assertEqual(prepared_run.resume_checkpoint.step, 4)
            self.assertEqual(prepared_run.resume_checkpoint.model_path, model_path)
            self.assertEqual(prepared_run.resume_checkpoint.optimizer_path, optimizer_path)

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
                        "hidden_size": 32,
                        "num_heads": 4,
                        "intermediate_size": 64,
                        "num_layers": 2,
                    },
                    "optimizer": {
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                    },
                    "training": {
                        "seed": 45,
                        "total_steps": 10,
                        "batch_size": 1,
                        "checkpoint_interval": 5,
                        "target_vocab_size": 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
