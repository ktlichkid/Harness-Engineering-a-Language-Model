import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import Parameter

from small_scale_llm.checkpointing import load_optimizer_checkpoint, save_optimizer_checkpoint
from small_scale_llm.optim import AdamW


class OptimizerCheckpointTests(unittest.TestCase):
    def _build_optimizer(self) -> tuple[Parameter, AdamW]:
        parameter = Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        optimizer = AdamW(
            [parameter],
            lr=0.05,
            betas=(0.9, 0.95),
            eps=1e-7,
            weight_decay=0.01,
        )
        return parameter, optimizer

    def test_save_optimizer_checkpoint_writes_state_dict(self) -> None:
        parameter, optimizer = self._build_optimizer()
        parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
        optimizer.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_optimizer_checkpoint(optimizer, Path(temp_dir) / "optimizer.pt")
            self.assertTrue(path.exists())
            loaded = torch.load(path, map_location="cpu")

        self.assertIn("state", loaded)
        self.assertIn("param_groups", loaded)

    def test_load_optimizer_checkpoint_restores_expected_optimizer_slots(self) -> None:
        source_parameter, source_optimizer = self._build_optimizer()
        source_parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
        source_optimizer.step()
        saved_state = source_optimizer.state_dict()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_optimizer_checkpoint(source_optimizer, Path(temp_dir) / "optimizer.pt")
            target_parameter, target_optimizer = self._build_optimizer()
            restored = load_optimizer_checkpoint(target_optimizer, path)

        self.assertEqual(restored["param_groups"], saved_state["param_groups"])
        self.assertEqual(restored["state"].keys(), saved_state["state"].keys())

        saved_entry = next(iter(saved_state["state"].values()))
        restored_entry = next(iter(target_optimizer.state.values()))
        self.assertEqual(restored_entry["step"], saved_entry["step"])
        self.assertTrue(torch.allclose(restored_entry["exp_avg"], saved_entry["exp_avg"]))
        self.assertTrue(torch.allclose(restored_entry["exp_avg_sq"], saved_entry["exp_avg_sq"]))

    def test_reloaded_optimizer_matches_expected_next_update(self) -> None:
        source_parameter, source_optimizer = self._build_optimizer()
        source_parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_optimizer_checkpoint(source_optimizer, Path(temp_dir) / "optimizer.pt")

            target_parameter, target_optimizer = self._build_optimizer()
            with torch.no_grad():
                target_parameter.copy_(source_parameter.detach())
            load_optimizer_checkpoint(target_optimizer, path)

        next_gradient = torch.tensor([0.1, -0.2], dtype=torch.float32)
        source_parameter.grad = next_gradient.clone()
        target_parameter.grad = next_gradient.clone()

        source_optimizer.step()
        target_optimizer.step()

        self.assertTrue(torch.allclose(target_parameter.detach(), source_parameter.detach()))


if __name__ == "__main__":
    unittest.main()
