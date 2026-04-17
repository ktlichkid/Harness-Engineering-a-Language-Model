import unittest

import torch
from torch.nn import Parameter

from small_scale_llm.optim import AdamW


class AdamWUpdateTests(unittest.TestCase):
    def test_single_step_matches_expected_parameter_update(self) -> None:
        parameter = Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = AdamW([parameter], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        parameter.grad = torch.tensor([0.5], dtype=torch.float32)

        optimizer.step()

        self.assertTrue(torch.allclose(parameter.detach(), torch.tensor([0.9])))

    def test_weight_decay_is_decoupled_from_gradient_moments(self) -> None:
        parameter = Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = AdamW([parameter], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        parameter.grad = torch.tensor([0.0], dtype=torch.float32)

        optimizer.step()

        self.assertTrue(torch.allclose(parameter.detach(), torch.tensor([0.99])))


class AdamWStateTests(unittest.TestCase):
    def test_optimizer_initializes_serializable_state_entries(self) -> None:
        parameter = Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        optimizer = AdamW([parameter], lr=0.05)
        parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)

        optimizer.step()
        state = optimizer.state[parameter]

        self.assertEqual(state["step"], 1)
        self.assertEqual(tuple(state["exp_avg"].shape), (2,))
        self.assertEqual(tuple(state["exp_avg_sq"].shape), (2,))

        serialized = optimizer.state_dict()
        self.assertIn("state", serialized)
        self.assertEqual(len(serialized["state"]), 1)
        entry = next(iter(serialized["state"].values()))
        self.assertEqual(entry["step"], 1)
        self.assertEqual(tuple(entry["exp_avg"].shape), (2,))
        self.assertEqual(tuple(entry["exp_avg_sq"].shape), (2,))

    def test_rejects_sparse_gradients(self) -> None:
        parameter = Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        optimizer = AdamW([parameter], lr=0.1)
        parameter.grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1]]),
            values=torch.tensor([1.0, -1.0]),
            size=(2,),
        )

        with self.assertRaisesRegex(RuntimeError, "does not support sparse gradients"):
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
