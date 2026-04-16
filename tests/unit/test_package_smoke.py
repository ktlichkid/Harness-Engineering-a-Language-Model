"""Baseline smoke tests for the initial package scaffold."""

from small_scale_llm import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
