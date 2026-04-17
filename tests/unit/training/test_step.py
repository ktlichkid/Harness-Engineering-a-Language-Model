import unittest

import torch

from small_scale_llm.model import TransformerLanguageModel
from small_scale_llm.optim import AdamW
from small_scale_llm.training import (
    prepare_language_model_batch,
    run_training_loop,
    run_training_step,
)


class BatchPreparationTests(unittest.TestCase):
    def test_prepare_language_model_batch_shifts_inputs_and_targets(self) -> None:
        token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)

        inputs, targets = prepare_language_model_batch(token_ids)

        self.assertTrue(torch.equal(inputs, torch.tensor([[1, 2, 3]], dtype=torch.int64)))
        self.assertTrue(torch.equal(targets, torch.tensor([[2, 3, 4]], dtype=torch.int64)))

    def test_prepare_language_model_batch_rejects_short_sequences(self) -> None:
        token_ids = torch.tensor([[1]], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "at least two positions"):
            prepare_language_model_batch(token_ids)


class TrainingStepTests(unittest.TestCase):
    def _build_model_and_optimizer(self) -> tuple[TransformerLanguageModel, AdamW]:
        torch.manual_seed(7)
        model = TransformerLanguageModel(
            vocab_size=16,
            max_sequence_length=8,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            num_layers=1,
        )
        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        return model, optimizer

    def test_run_training_step_updates_parameters_and_returns_debug_logs(self) -> None:
        model, optimizer = self._build_model_and_optimizer()
        token_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.int64)
        before = model.output_weight.detach().clone()

        logs = run_training_step(model, optimizer, token_ids, step_index=3)

        self.assertEqual(logs["step"], 3)
        self.assertEqual(logs["tokens"], 6)
        self.assertGreater(logs["loss"], 0.0)
        self.assertGreater(logs["grad_norm"], 0.0)
        self.assertIn("mean_logit", logs)
        self.assertFalse(torch.equal(before, model.output_weight.detach()))

    def test_run_training_loop_collects_step_logs(self) -> None:
        model, optimizer = self._build_model_and_optimizer()
        batches = [
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            torch.tensor([[2, 3, 4, 5]], dtype=torch.int64),
        ]

        logs = run_training_loop(model, optimizer, batches)

        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["step"], 0)
        self.assertEqual(logs[1]["step"], 1)
        self.assertTrue(all(entry["tokens"] == 3 for entry in logs))


if __name__ == "__main__":
    unittest.main()
