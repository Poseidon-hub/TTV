from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

