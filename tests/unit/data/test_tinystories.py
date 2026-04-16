import tempfile
import unittest
from pathlib import Path

from small_scale_llm.data import (
    iter_tinystories_records,
    load_tinystories_config,
    materialize_tinystories_split,
)


class TinyStoriesConfigTests(unittest.TestCase):
    def test_load_tinystories_config_uses_checked_in_split_metadata(self) -> None:
        config = load_tinystories_config("configs/milestone1/tinystories.json", "train")

        self.assertEqual(config.split, "train")
        self.assertEqual(config.filename, "TinyStories-train.txt")
        self.assertEqual(config.local_path, Path("data/tinystories/TinyStories-train.txt"))
        self.assertIn(
            "roneneldan/TinyStories/resolve/main/TinyStories-train.txt", config.source_url
        )

    def test_materialize_tinystories_split_uses_existing_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TinyStories-valid.txt"
            cache_file.write_text("cached story\n", encoding="utf-8")
            config = load_tinystories_config("configs/milestone1/tinystories.json", "validation")
            config = type(config)(
                split=config.split,
                cache_dir=Path(temp_dir),
                filename=config.filename,
                source_url=config.source_url,
            )

            resolved_path = materialize_tinystories_split(config, download=False)

            self.assertEqual(resolved_path, cache_file)


class TinyStoriesRecordTests(unittest.TestCase):
    def test_iter_tinystories_records_uses_story_separator_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "TinyStories-train.txt"
            dataset_path.write_text(
                (
                    "First line of story one.\n"
                    "Second line of story one.\n"
                    "<|endoftext|>\n"
                    "Story two starts here.\n"
                    "<|endoftext|>\n"
                ),
                encoding="utf-8",
            )

            records = list(iter_tinystories_records(dataset_path, "train"))

        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0].as_dict(),
            {
                "text": "First line of story one.\nSecond line of story one.",
                "split": "train",
                "record_index": 0,
                "source_path": str(dataset_path),
            },
        )
        self.assertEqual(records[1].record_index, 1)
        self.assertEqual(records[1].text, "Story two starts here.")


if __name__ == "__main__":
    unittest.main()
