"""OpenWebText dataset loading with a narrow, reproducible data contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class OpenWebTextConfig:
    """Configuration for reproducible OpenWebText loading."""

    dataset_id: str
    config_name: str
    split: str
    cache_dir: Path
    filename: str
    rows_api_url: str
    page_size: int
    default_max_records: int

    @property
    def local_path(self) -> Path:
        return self.cache_dir / self.filename


@dataclass(frozen=True)
class OpenWebTextRecord:
    """Normalized OpenWebText record shape for tokenizer or trainer inputs."""

    text: str
    split: str
    record_index: int
    source_path: str

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "split": self.split,
            "record_index": self.record_index,
            "source_path": self.source_path,
        }


def load_openwebtext_config(config_path: str | Path) -> OpenWebTextConfig:
    """Load OpenWebText settings from a checked-in JSON config."""

    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return OpenWebTextConfig(
        dataset_id=config_data["dataset_id"],
        config_name=config_data["config_name"],
        split=config_data["split"],
        cache_dir=Path(config_data["cache_dir"]),
        filename=config_data["filename"],
        rows_api_url=config_data["rows_api_url"],
        page_size=config_data["page_size"],
        default_max_records=config_data["default_max_records"],
    )


def materialize_openwebtext_split(
    config: OpenWebTextConfig,
    *,
    max_records: int | None = None,
    download: bool = True,
) -> Path:
    """
    Resolve a local OpenWebText cache file.

    If the cached file is missing and ``download`` is true, fetch records from
    the configured rows API and store them as JSON Lines with one ``text`` field
    per line.
    """

    local_path = config.local_path
    if local_path.exists():
        return local_path

    if not download:
        raise FileNotFoundError(
            f"OpenWebText split '{config.split}' is not available at {local_path}."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    target_records = config.default_max_records if max_records is None else max_records
    remaining = target_records
    offset = 0

    with local_path.open("w", encoding="utf-8") as handle:
        while remaining > 0:
            page_length = min(config.page_size, remaining)
            query = urlencode(
                {
                    "dataset": config.dataset_id,
                    "config": config.config_name,
                    "split": config.split,
                    "offset": offset,
                    "length": page_length,
                }
            )
            with urlopen(f"{config.rows_api_url}?{query}", timeout=60) as response:
                payload = json.load(response)

            rows = payload["rows"]
            if not rows:
                break

            for row in rows:
                handle.write(json.dumps({"text": row["row"]["text"]}, ensure_ascii=False))
                handle.write("\n")

            fetched = len(rows)
            offset += fetched
            remaining -= fetched

    return local_path


def iter_openwebtext_records(dataset_path: str | Path, split: str) -> Iterator[OpenWebTextRecord]:
    """Yield normalized OpenWebText records from a cached JSONL file."""

    source_path = Path(dataset_path)
    with source_path.open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            payload = json.loads(line)
            text = payload["text"].strip()
            if not text:
                continue
            yield OpenWebTextRecord(
                text=text,
                split=split,
                record_index=record_index,
                source_path=str(source_path),
            )
