import math
import unittest

import torch

from small_scale_llm.model import cross_entropy_loss


class CrossEntropyLossTests(unittest.TestCase):
    def test_returns_expected_mean_loss_for_representative_logits(self) -> None:
        logits = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, 1]], dtype=torch.int64)

        loss = cross_entropy_loss(logits, targets)
        expected = math.log(math.exp(2.0) + 1.0) - 2.0

        self.assertTrue(torch.allclose(loss, torch.tensor(expected, dtype=torch.float32)))

    def test_supports_none_reduction_with_ignore_index(self) -> None:
        logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, -100]], dtype=torch.int64)

        loss = cross_entropy_loss(logits, targets, ignore_index=-100, reduction="none")
        expected_first = torch.logsumexp(logits[0, 0], dim=-1) - logits[0, 0, 0]

        self.assertEqual(loss.shape, (1, 2))
        self.assertTrue(torch.allclose(loss[0, 0], expected_first))
        self.assertEqual(loss[0, 1].item(), 0.0)

    def test_supports_sum_reduction(self) -> None:
        logits = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, 0]], dtype=torch.int64)

        loss = cross_entropy_loss(logits, targets, reduction="sum")
        expected = (
            torch.logsumexp(logits[0, 0], dim=-1)
            - logits[0, 0, 0]
            + torch.logsumexp(logits[0, 1], dim=-1)
            - logits[0, 1, 0]
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_returns_zero_mean_when_all_positions_are_ignored(self) -> None:
        logits = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]], dtype=torch.float32)
        targets = torch.tensor([[-1, -1]], dtype=torch.int64)

        loss = cross_entropy_loss(logits, targets, ignore_index=-1)

        self.assertEqual(loss.item(), 0.0)

    def test_rejects_targets_outside_vocabulary(self) -> None:
        logits = torch.ones((1, 2, 3), dtype=torch.float32)
        targets = torch.tensor([[0, 3]], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "outside the logits vocabulary"):
            cross_entropy_loss(logits, targets)


if __name__ == "__main__":
    unittest.main()
