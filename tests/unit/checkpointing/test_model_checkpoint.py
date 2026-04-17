import tempfile
import unittest
from pathlib import Path

import torch

from small_scale_llm.checkpointing import load_model_checkpoint, save_model_checkpoint
from small_scale_llm.model import TransformerLanguageModel


class ModelCheckpointTests(unittest.TestCase):
    def _build_model(self) -> TransformerLanguageModel:
        return TransformerLanguageModel(
            vocab_size=9,
            max_sequence_length=4,
            hidden_size=4,
            num_heads=2,
            intermediate_size=6,
            num_layers=1,
        )

    def test_save_and_load_restore_model_parameters(self) -> None:
        source_model = self._build_model()
        with torch.no_grad():
            source_model.output_bias.fill_(1.25)
            source_model.blocks[0].attention.query_weight.fill_(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = save_model_checkpoint(
                source_model,
                Path(temp_dir) / "model-checkpoint.pt",
            )

            restored_model = self._build_model()
            with torch.no_grad():
                restored_model.output_bias.zero_()
                restored_model.blocks[0].attention.query_weight.zero_()

            loaded_keys = load_model_checkpoint(restored_model, checkpoint_path)

        self.assertIn("output_bias", loaded_keys)
        self.assertTrue(torch.equal(restored_model.output_bias, source_model.output_bias))
        self.assertTrue(
            torch.equal(
                restored_model.blocks[0].attention.query_weight,
                source_model.blocks[0].attention.query_weight,
            )
        )

    def test_saved_checkpoint_matches_state_dict_keys_deterministically(self) -> None:
        model = self._build_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = save_model_checkpoint(model, Path(temp_dir) / "model.pt")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(sorted(payload.keys()), sorted(model.state_dict().keys()))


if __name__ == "__main__":
    unittest.main()
