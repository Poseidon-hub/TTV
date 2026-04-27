from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils import load_json, save_json


class CharTokenizer:
    """Простой токенизатор по символам."""

    def __init__(self, stoi: Dict[str, int] | None = None, itos: List[str] | None = None):
        self.stoi: Dict[str, int] = stoi or {}
        self.itos: List[str] = itos or []

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def build_from_text(self, text: str) -> "CharTokenizer":
        unique_chars = sorted(set(text))
        if " " not in unique_chars:
            unique_chars.insert(0, " ")

        self.itos = unique_chars
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}
        return self

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        fallback_id = self.stoi.get(" ")
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif fallback_id is not None:
                # Неизвестные символы заменяем пробелом.
                ids.append(fallback_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars: List[str] = []
        for idx in ids:
            int_idx = int(idx)
            if 0 <= int_idx < len(self.itos):
                chars.append(self.itos[int_idx])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        save_json({"stoi": self.stoi, "itos": self.itos}, path)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = load_json(path)
        return cls(stoi=data["stoi"], itos=data["itos"])

