import json
import tempfile
import unittest
from pathlib import Path

from small_scale_llm.tokenizer import (
    train_bpe_from_texts,
    train_bpe_from_tinystories,
    write_bpe_artifact,
)


class BPETrainingTests(unittest.TestCase):
    def test_train_bpe_from_texts_is_deterministic(self) -> None:
        texts = ["aaaa aaaa", "aaaa"]

        first = train_bpe_from_texts(texts, target_vocab_size=6)
        second = train_bpe_from_texts(texts, target_vocab_size=6)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.merges[0], ["a", "a"])
        self.assertIn("aaaa</w>", first.vocab)

    def test_train_bpe_from_tinystories_uses_story_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "TinyStories-valid.txt"
            dataset_path.write_text(
                ("alpha beta\nline two\n<|endoftext|>\nalpha gamma\n<|endoftext|>\n"),
                encoding="utf-8",
            )

            artifact = train_bpe_from_tinystories(
                dataset_path,
                split="validation",
                target_vocab_size=20,
            )

        self.assertGreaterEqual(len(artifact.vocab), 1)
        self.assertGreaterEqual(len(artifact.merges), 1)
        self.assertIn("alpha</w>", artifact.token_to_id)

    def test_write_bpe_artifact_writes_serializable_json(self) -> None:
        artifact = train_bpe_from_texts(["abab abab"], target_vocab_size=10)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_bpe_artifact(artifact, Path(temp_dir) / "bpe.json")
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["merges"], artifact.merges)
        self.assertEqual(saved["vocab"], artifact.vocab)
        self.assertEqual(saved["token_to_id"], artifact.token_to_id)


if __name__ == "__main__":
    unittest.main()
