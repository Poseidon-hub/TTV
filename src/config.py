from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from src.utils import load_json, save_json


@dataclass
class ProjectConfig:
    """Общая конфигурация модели и обучения."""

    vocab_size: int = 0
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    batch_size: int = 32
    max_iters: int = 800
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 3e-4
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load_json(cls, path: str | Path) -> "ProjectConfig":
        return cls.from_dict(load_json(path))

