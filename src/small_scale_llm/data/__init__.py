"""Dataset ingestion interfaces for Milestone 1."""

from .tinystories import (
    TinyStoriesConfig,
    TinyStoryRecord,
    iter_tinystories_records,
    load_tinystories_config,
    materialize_tinystories_split,
)

__all__ = [
    "TinyStoriesConfig",
    "TinyStoryRecord",
    "iter_tinystories_records",
    "load_tinystories_config",
    "materialize_tinystories_split",
]
