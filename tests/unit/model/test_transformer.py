import unittest

import torch

from small_scale_llm.model import CausalSelfAttention, TransformerBlock, TransformerLanguageModel


class CausalSelfAttentionTests(unittest.TestCase):
    def test_attention_returns_hidden_shaped_output(self) -> None:
        attention = CausalSelfAttention(hidden_size=4, num_heads=2)
        hidden_states = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )

        output = attention(hidden_states)

        self.assertEqual(tuple(output.shape), (1, 2, 4))

    def test_attention_rejects_hidden_size_not_divisible_by_num_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            CausalSelfAttention(hidden_size=5, num_heads=2)


class TransformerBlockTests(unittest.TestCase):
    def test_transformer_block_preserves_sequence_hidden_shape(self) -> None:
        block = TransformerBlock(hidden_size=6, num_heads=2, intermediate_size=9)
        hidden_states = torch.randn(2, 3, 6, dtype=torch.float32)

        output = block(hidden_states)

        self.assertEqual(tuple(output.shape), (2, 3, 6))


class TransformerLanguageModelTests(unittest.TestCase):
    def test_language_model_returns_vocab_sized_logits(self) -> None:
        model = TransformerLanguageModel(
            vocab_size=11,
            max_sequence_length=5,
            hidden_size=6,
            num_heads=2,
            intermediate_size=8,
            num_layers=2,
        )
        token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)

        logits = model(token_ids)

        self.assertEqual(tuple(logits.shape), (2, 3, 11))

    def test_language_model_registers_block_and_logits_parameters(self) -> None:
        model = TransformerLanguageModel(
            vocab_size=7,
            max_sequence_length=4,
            hidden_size=4,
            num_heads=2,
            intermediate_size=6,
            num_layers=1,
        )

        parameter_names = {name for name, _ in model.named_parameters()}

        self.assertIn("blocks.0.attention.query_weight", parameter_names)
        self.assertIn("blocks.0.feedforward.input_weight", parameter_names)
        self.assertIn("embedding.token_embedding.weight", parameter_names)
        self.assertIn("final_norm.weight", parameter_names)
        self.assertIn("output_weight", parameter_names)
        self.assertIn("output_bias", parameter_names)

    def test_language_model_rejects_non_positive_layer_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_layers must be positive"):
            TransformerLanguageModel(
                vocab_size=7,
                max_sequence_length=4,
                hidden_size=4,
                num_heads=2,
                intermediate_size=6,
                num_layers=0,
            )


if __name__ == "__main__":
    unittest.main()
