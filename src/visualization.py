from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _display_char(ch: str) -> str:
    if ch == " ":
        return "[sp]"
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    return ch


def build_attention_heatmap(
    chars: Sequence[str],
    matrix: np.ndarray,
    save_path: str | Path | None = None,
    step: int | None = None,
    generated_char: str = "",
):
    if len(chars) == 0 or matrix.size == 0:
        chars = [" "]
        matrix = np.zeros((1, 1), dtype=np.float32)

    visual_size = max(6.0, min(14.0, len(chars) * 0.4))
    fig, ax = plt.subplots(figsize=(visual_size, visual_size))

    image = ax.imshow(matrix, cmap="magma", interpolation="nearest")
    labels = [_display_char(ch) for ch in chars]

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    title = "Attention map"
    if step is not None:
        title = f"{title} — step {step}"
    if generated_char:
        title = f"{title} — generated: '{_display_char(generated_char)}'"

    ax.set_title(title)
    ax.set_xlabel("Символы по оси X")
    ax.set_ylabel("Символы по оси Y")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_obj, dpi=160, bbox_inches="tight")

    return fig
