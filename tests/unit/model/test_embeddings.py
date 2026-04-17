import unittest

import torch

from small_scale_llm.model import PositionEmbedding, TokenEmbedding, TokenPositionEmbedding


class TokenEmbeddingTests(unittest.TestCase):
    def test_token_embedding_returns_expected_shape(self) -> None:
        embedding = TokenEmbedding(vocab_size=8, embedding_dim=4)
        token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)

        embedded = embedding(token_ids)

        self.assertEqual(tuple(embedded.shape), (2, 3, 4))

    def test_token_embedding_rejects_out_of_range_ids(self) -> None:
        embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)
        token_ids = torch.tensor([[0, 4]], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "outside the embedding vocabulary"):
            embedding(token_ids)


class PositionEmbeddingTests(unittest.TestCase):
    def test_position_embedding_returns_batch_aligned_shape(self) -> None:
        embedding = PositionEmbedding(max_sequence_length=6, embedding_dim=5)
        token_ids = torch.tensor([[7, 8, 9], [1, 2, 3]], dtype=torch.int64)

        positions = embedding(token_ids)

        self.assertEqual(tuple(positions.shape), (2, 3, 5))
        self.assertTrue(torch.equal(positions[0], positions[1]))

    def test_position_embedding_rejects_sequence_overflow(self) -> None:
        embedding = PositionEmbedding(max_sequence_length=2, embedding_dim=4)
        token_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "exceeds the configured maximum"):
            embedding(token_ids)


class TokenPositionEmbeddingTests(unittest.TestCase):
    def test_combined_embedding_matches_component_sum(self) -> None:
        module = TokenPositionEmbedding(vocab_size=10, max_sequence_length=4, embedding_dim=3)
        token_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)

        combined = module(token_ids)
        expected = module.token_embedding(token_ids) + module.position_embedding(token_ids)

        self.assertTrue(torch.allclose(combined, expected))


if __name__ == "__main__":
    unittest.main()
