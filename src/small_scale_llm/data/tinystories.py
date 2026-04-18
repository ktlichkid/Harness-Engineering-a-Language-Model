"""TinyStories dataset loading with a narrow, reproducible data contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

TINYSTORIES_RECORD_SEPARATOR = "<|endoftext|>"


@dataclass(frozen=True)
class TinyStoriesConfig:
    """Configuration for one TinyStories split."""

    split: str
    cache_dir: Path
    filename: str
    source_url: str

    @property
    def local_path(self) -> Path:
        return self.cache_dir / self.filename


@dataclass(frozen=True)
class TinyStoryRecord:
    """Normalized record shape for downstream tokenizer or trainer inputs."""

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


def load_tinystories_config(config_path: str | Path, split: str) -> TinyStoriesConfig:
    """Load TinyStories split settings from a checked-in JSON config."""

    config_data = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    split_data = config_data["splits"][split]
    return TinyStoriesConfig(
        split=split,
        cache_dir=Path(config_data["cache_dir"]),
        filename=split_data["filename"],
        source_url=split_data["source_url"],
    )


def materialize_tinystories_split(
    config: TinyStoriesConfig,
    *,
    download: bool = True,
) -> Path:
    """
    Resolve a local TinyStories split path.

    If the cached file is missing and ``download`` is true, fetch the official
    split file into the configured cache directory.
    """

    local_path = config.local_path
    if local_path.exists():
        return local_path

    if not download:
        raise FileNotFoundError(
            f"TinyStories split '{config.split}' is not available at {local_path}."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(config.source_url, local_path)
    return local_path


def iter_tinystories_records(dataset_path: str | Path, split: str) -> Iterator[TinyStoryRecord]:
    """
    Yield normalized TinyStories records.

    The current contract is one story per ``<|endoftext|>`` boundary with:
    ``text``, ``split``, ``record_index``, and ``source_path``.
    """

    source_path = Path(dataset_path)
    raw_text = source_path.read_text(encoding="utf-8")
    for record_index, chunk in enumerate(raw_text.split(TINYSTORIES_RECORD_SEPARATOR)):
        story_text = chunk.strip()
        if not story_text:
            continue
        yield TinyStoryRecord(
            text=story_text,
            split=split,
            record_index=record_index,
            source_path=str(source_path),
        )
