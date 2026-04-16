"""Deterministic BPE vocabulary training for Milestone 1."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from small_scale_llm.data import iter_tinystories_records

END_OF_WORD = "</w>"


@dataclass(frozen=True)
class BPEArtifact:
    """Serializable BPE vocabulary training output."""

    vocab: list[str]
    merges: list[list[str]]
    token_to_id: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }


def _split_words(text: str) -> list[tuple[str, ...]]:
    words = []
    for raw_word in text.split():
        symbols = tuple(list(raw_word) + [END_OF_WORD])
        if symbols:
            words.append(symbols)
    return words


def _build_word_frequencies(texts: list[str]) -> Counter[tuple[str, ...]]:
    frequencies: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        frequencies.update(_split_words(text))
    return frequencies


def _count_pairs(
    word_frequencies: Counter[tuple[str, ...]],
) -> Counter[tuple[str, str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in word_frequencies.items():
        for index in range(len(symbols) - 1):
            pair_counts[(symbols[index], symbols[index + 1])] += frequency
    return pair_counts


def _merge_symbols(
    word_frequencies: Counter[tuple[str, ...]],
    pair_to_merge: tuple[str, str],
) -> Counter[tuple[str, ...]]:
    merged_token = "".join(pair_to_merge)
    merged_frequencies: Counter[tuple[str, ...]] = Counter()

    for symbols, frequency in word_frequencies.items():
        merged_symbols: list[str] = []
        index = 0
        while index < len(symbols):
            if (
                index < len(symbols) - 1
                and symbols[index] == pair_to_merge[0]
                and symbols[index + 1] == pair_to_merge[1]
            ):
                merged_symbols.append(merged_token)
                index += 2
                continue
            merged_symbols.append(symbols[index])
            index += 1
        merged_frequencies[tuple(merged_symbols)] += frequency

    return merged_frequencies


def _build_vocab(word_frequencies: Counter[tuple[str, ...]]) -> list[str]:
    vocab = {symbol for word in word_frequencies for symbol in word}
    return sorted(vocab)


def train_bpe_from_texts(
    texts: list[str],
    *,
    target_vocab_size: int,
    min_pair_count: int = 1,
) -> BPEArtifact:
    """Train BPE merges from text samples and return a serializable artifact."""

    word_frequencies = _build_word_frequencies(texts)
    merges: list[list[str]] = []

    while True:
        vocab = _build_vocab(word_frequencies)
        if len(vocab) >= target_vocab_size:
            break

        pair_counts = _count_pairs(word_frequencies)
        if not pair_counts:
            break

        pair_to_merge, pair_count = min(
            pair_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if pair_count < min_pair_count:
            break

        merges.append([pair_to_merge[0], pair_to_merge[1]])
        word_frequencies = _merge_symbols(word_frequencies, pair_to_merge)

    vocab = _build_vocab(word_frequencies)
    token_to_id = {token: index for index, token in enumerate(vocab)}
    return BPEArtifact(vocab=vocab, merges=merges, token_to_id=token_to_id)


def train_bpe_from_tinystories(
    dataset_path: str | Path,
    *,
    split: str,
    target_vocab_size: int,
    min_pair_count: int = 1,
) -> BPEArtifact:
    """Train BPE merges from the normalized TinyStories ingestion path."""

    texts = [record.text for record in iter_tinystories_records(dataset_path, split)]
    return train_bpe_from_texts(
        texts,
        target_vocab_size=target_vocab_size,
        min_pair_count=min_pair_count,
    )


def write_bpe_artifact(artifact: BPEArtifact, output_path: str | Path) -> Path:
    """Write a BPE artifact to a JSON file."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
    return destination
