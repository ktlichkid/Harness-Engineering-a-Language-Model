"""Dataset ingestion interfaces for Milestone 1."""

from .openwebtext import (
    OpenWebTextConfig,
    OpenWebTextRecord,
    iter_openwebtext_records,
    load_openwebtext_config,
    materialize_openwebtext_split,
)
from .tinystories import (
    TinyStoriesConfig,
    TinyStoryRecord,
    iter_tinystories_records,
    load_tinystories_config,
    materialize_tinystories_split,
)

__all__ = [
    "OpenWebTextConfig",
    "OpenWebTextRecord",
    "iter_openwebtext_records",
    "load_openwebtext_config",
    "materialize_openwebtext_split",
    "TinyStoriesConfig",
    "TinyStoryRecord",
    "iter_tinystories_records",
    "load_tinystories_config",
    "materialize_tinystories_split",
]
