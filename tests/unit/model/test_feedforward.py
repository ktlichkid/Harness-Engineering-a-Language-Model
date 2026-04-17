import unittest

import torch

from small_scale_llm.model import FeedForwardNetwork, LayerNorm, gelu


class FeedForwardNetworkTests(unittest.TestCase):
    def test_feedforward_returns_expected_shape(self) -> None:
        module = FeedForwardNetwork(hidden_size=4, intermediate_size=6)
        hidden_states = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )

        output = module(hidden_states)

        self.assertEqual(output.shape, (1, 2, 4))

    def test_feedforward_parameters_are_registered(self) -> None:
        module = FeedForwardNetwork(hidden_size=3, intermediate_size=5)

        parameter_names = {name for name, _ in module.named_parameters()}

        self.assertEqual(
            parameter_names,
            {"input_bias", "input_weight", "output_bias", "output_weight"},
        )

    def test_feedforward_matches_manual_projection_path(self) -> None:
        module = FeedForwardNetwork(hidden_size=2, intermediate_size=3)
        with torch.no_grad():
            module.input_weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]]))
            module.input_bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
            module.output_weight.copy_(torch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 1.5]]))
            module.output_bias.copy_(torch.tensor([0.25, -0.5]))

        hidden_states = torch.tensor([[[2.0, -1.0]]], dtype=torch.float32)
        hidden_projection = hidden_states @ module.input_weight + module.input_bias
        expected = gelu(hidden_projection) @ module.output_weight + module.output_bias

        output = module(hidden_states)

        self.assertTrue(torch.allclose(output, expected))


class LayerNormTests(unittest.TestCase):
    def test_layer_norm_preserves_input_shape(self) -> None:
        layer_norm = LayerNorm(hidden_size=4)
        hidden_states = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )

        normalized = layer_norm(hidden_states)

        self.assertEqual(normalized.shape, (1, 2, 4))

    def test_layer_norm_produces_zero_mean_unit_variance_with_default_affine(self) -> None:
        layer_norm = LayerNorm(hidden_size=4)
        hidden_states = torch.tensor(
            [[[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]],
            dtype=torch.float32,
        )

        normalized = layer_norm(hidden_states)
        means = normalized.mean(dim=-1)
        variances = normalized.var(dim=-1, unbiased=False)

        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))
        self.assertTrue(torch.allclose(variances, torch.ones_like(variances), atol=1e-4))

    def test_layer_norm_rejects_hidden_width_mismatch(self) -> None:
        layer_norm = LayerNorm(hidden_size=3)
        hidden_states = torch.ones((1, 2, 4), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "hidden state width must match hidden_size"):
            layer_norm(hidden_states)


if __name__ == "__main__":
    unittest.main()
