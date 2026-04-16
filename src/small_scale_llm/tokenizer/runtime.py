"""Runtime BPE encode/decode behavior for Milestone 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .bpe_training import BPEArtifact, END_OF_WORD


@dataclass(frozen=True)
class BPETokenizer:
    """Runtime tokenizer backed by the current BPE training artifact."""

    vocab: list[str]
    merges: list[list[str]]
    token_to_id: dict[str, int]

    @classmethod
    def from_artifact(cls, artifact: BPEArtifact) -> "BPETokenizer":
        return cls(
            vocab=list(artifact.vocab),
            merges=[list(merge) for merge in artifact.merges],
            token_to_id=dict(artifact.token_to_id),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in text.split():
            symbols = list(word) + [END_OF_WORD]
            merged_symbols = self._apply_merges(symbols)
            for symbol in merged_symbols:
                if symbol not in self.token_to_id:
                    raise ValueError(
                        f"Token '{symbol}' is not present in the tokenizer vocabulary."
                    )
                token_ids.append(self.token_to_id[symbol])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        pieces: list[str] = []
        current_word: list[str] = []

        for token_id in token_ids:
            if token_id not in id_to_token:
                raise ValueError(f"Token id '{token_id}' is not present in the tokenizer state.")
            token = id_to_token[token_id]
            if token.endswith(END_OF_WORD):
                current_word.append(token[: -len(END_OF_WORD)])
                pieces.append("".join(current_word))
                current_word = []
                continue
            current_word.append(token)

        if current_word:
            pieces.append("".join(current_word))

        return " ".join(piece for piece in pieces if piece)

    def save(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        merged_symbols = list(symbols)
        for left, right in self.merges:
            merged_token = left + right
            next_symbols: list[str] = []
            index = 0
            while index < len(merged_symbols):
                if (
                    index < len(merged_symbols) - 1
                    and merged_symbols[index] == left
                    and merged_symbols[index + 1] == right
                ):
                    next_symbols.append(merged_token)
                    index += 2
                    continue
                next_symbols.append(merged_symbols[index])
                index += 1
            merged_symbols = next_symbols
        return merged_symbols


def load_bpe_tokenizer(input_path: str | Path) -> BPETokenizer:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return BPETokenizer(
        vocab=list(data["vocab"]),
        merges=[list(merge) for merge in data["merges"]],
        token_to_id=dict(data["token_to_id"]),
    )
