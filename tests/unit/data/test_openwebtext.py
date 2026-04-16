import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from small_scale_llm.data import (
    iter_openwebtext_records,
    load_openwebtext_config,
    materialize_openwebtext_split,
)


class OpenWebTextConfigTests(unittest.TestCase):
    def test_load_openwebtext_config_uses_checked_in_metadata(self) -> None:
        config = load_openwebtext_config("configs/milestone1/openwebtext.json")

        self.assertEqual(config.dataset_id, "Skylion007/openwebtext")
        self.assertEqual(config.config_name, "plain_text")
        self.assertEqual(config.split, "train")
        self.assertEqual(config.local_path, Path("data/openwebtext/openwebtext-train.jsonl"))
        self.assertEqual(config.page_size, 100)
        self.assertEqual(config.default_max_records, 100)

    def test_materialize_openwebtext_split_uses_existing_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "openwebtext-train.jsonl"
            cache_file.write_text('{"text": "cached text"}\n', encoding="utf-8")
            config = load_openwebtext_config("configs/milestone1/openwebtext.json")
            config = type(config)(
                dataset_id=config.dataset_id,
                config_name=config.config_name,
                split=config.split,
                cache_dir=Path(temp_dir),
                filename=config.filename,
                rows_api_url=config.rows_api_url,
                page_size=config.page_size,
                default_max_records=config.default_max_records,
            )

            resolved_path = materialize_openwebtext_split(config, download=False)

            self.assertEqual(resolved_path, cache_file)


class OpenWebTextRecordTests(unittest.TestCase):
    def test_materialize_openwebtext_split_writes_reproducible_jsonl_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_openwebtext_config("configs/milestone1/openwebtext.json")
            config = type(config)(
                dataset_id=config.dataset_id,
                config_name=config.config_name,
                split=config.split,
                cache_dir=Path(temp_dir),
                filename=config.filename,
                rows_api_url=config.rows_api_url,
                page_size=2,
                default_max_records=2,
            )
            payload = {
                "rows": [
                    {"row": {"text": "first openwebtext document"}},
                    {"row": {"text": "second openwebtext document"}},
                ]
            }

            with patch("small_scale_llm.data.openwebtext.urlopen", return_value=io.StringIO(json.dumps(payload))):
                dataset_path = materialize_openwebtext_split(config, download=True)

            written_lines = dataset_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                written_lines,
                [
                    '{"text": "first openwebtext document"}',
                    '{"text": "second openwebtext document"}',
                ],
            )

    def test_iter_openwebtext_records_yields_normalized_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "openwebtext-train.jsonl"
            dataset_path.write_text(
                (
                    '{"text": " First document text. "}\n'
                    '{"text": "Second document text."}\n'
                ),
                encoding="utf-8",
            )

            records = list(iter_openwebtext_records(dataset_path, "train"))

        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0].as_dict(),
            {
                "text": "First document text.",
                "split": "train",
                "record_index": 0,
                "source_path": str(dataset_path),
            },
        )
        self.assertEqual(records[1].record_index, 1)
        self.assertEqual(records[1].text, "Second document text.")


if __name__ == "__main__":
    unittest.main()
