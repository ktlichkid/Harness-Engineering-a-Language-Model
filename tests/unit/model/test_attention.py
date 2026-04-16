import unittest

import torch

from small_scale_llm.model import (
    build_causal_attention_mask,
    merge_attention_heads,
    project_attention_inputs,
    project_attention_output,
    scaled_dot_product_attention,
    split_attention_heads,
)


class AttentionShapeTests(unittest.TestCase):
    def test_split_and_merge_attention_heads_round_trip(self) -> None:
        hidden_states = torch.arange(24, dtype=torch.float32).view(1, 3, 8)

        split = split_attention_heads(hidden_states, num_heads=2)
        merged = merge_attention_heads(split)

        self.assertEqual(split.shape, (1, 2, 3, 4))
        self.assertTrue(torch.equal(merged, hidden_states))

    def test_projection_helpers_preserve_expected_tensor_contracts(self) -> None:
        hidden_states = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )
        identity = torch.eye(4, dtype=torch.float32)

        query, key, value = project_attention_inputs(
            hidden_states,
            num_heads=2,
            query_weight=identity,
            key_weight=identity,
            value_weight=identity,
        )
        projected = project_attention_output(value, output_weight=identity)

        self.assertEqual(query.shape, (1, 2, 2, 2))
        self.assertEqual(key.shape, (1, 2, 2, 2))
        self.assertEqual(value.shape, (1, 2, 2, 2))
        self.assertTrue(torch.equal(projected, hidden_states))


class AttentionMaskingTests(unittest.TestCase):
    def test_scaled_dot_product_attention_respects_causal_mask(self) -> None:
        query = torch.tensor([[[[1.0], [1.0], [1.0]]]], dtype=torch.float32)
        key = torch.tensor([[[[1.0], [1.0], [1.0]]]], dtype=torch.float32)
        value = torch.tensor([[[[10.0], [20.0], [30.0]]]], dtype=torch.float32)

        context, weights = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=build_causal_attention_mask(sequence_length=3),
        )

        self.assertEqual(context.shape, (1, 1, 3, 1))
        self.assertEqual(weights.shape, (1, 1, 3, 3))
        self.assertTrue(torch.allclose(weights[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(weights[0, 0, 1], torch.tensor([0.5, 0.5, 0.0])))
        self.assertTrue(torch.allclose(weights[0, 0, 2], torch.tensor([1 / 3, 1 / 3, 1 / 3])))
        self.assertTrue(torch.allclose(context[0, 0, :, 0], torch.tensor([10.0, 15.0, 20.0])))


if __name__ == "__main__":
    unittest.main()
