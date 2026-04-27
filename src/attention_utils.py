from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.generate import load_model_and_tokenizer


def prepare_attention_matrix(attentions: Sequence[torch.Tensor], layer_idx: int, head_idx: int) -> np.ndarray:
    if not attentions:
        raise ValueError("Список attention-карт пуст.")

    if layer_idx < 0 or layer_idx >= len(attentions):
        raise ValueError(f"Некорректный layer_idx={layer_idx}. Доступно слоёв: {len(attentions)}.")

    layer_attention = attentions[layer_idx]  # (B, heads, T, T)
    if head_idx < 0 or head_idx >= layer_attention.size(1):
        raise ValueError(f"Некорректный head_idx={head_idx}. Доступно голов: {layer_attention.size(1)}.")

    matrix = layer_attention[0, head_idx].detach().cpu().numpy()
    return matrix


def get_last_context_text(text: str, block_size: int) -> str:
    if block_size <= 0:
        return text
    if len(text) <= block_size:
        return text
    return text[-block_size:]


def prepare_attention_from_step(
    attention_step: Dict[str, Any],
    layer_idx: int = 0,
    head_idx: int = 0,
) -> Tuple[List[str], np.ndarray]:
    attentions = attention_step.get("attentions")
    if attentions is None:
        raise ValueError("В шаге генерации отсутствует ключ 'attentions'.")

    matrix = prepare_attention_matrix(attentions, layer_idx=layer_idx, head_idx=head_idx)

    step_text = str(attention_step.get("text", ""))
    context_text = get_last_context_text(step_text, matrix.shape[-1])
    chars = list(context_text)

    if len(chars) > matrix.shape[-1]:
        chars = chars[-matrix.shape[-1] :]
    elif len(chars) < matrix.shape[-1]:
        chars = ([" "] * (matrix.shape[-1] - len(chars))) + chars

    return chars, matrix


def get_attention_for_text(text: str, layer_idx: int = 0, head_idx: int = 0) -> Tuple[List[str], np.ndarray]:
    model, tokenizer, config, device = load_model_and_tokenizer()

    encoded = tokenizer.encode(text)
    if not encoded:
        encoded = [tokenizer.stoi.get(" ", 0)]

    encoded = encoded[-config.block_size :]
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        _, _, attentions = model(idx, return_attentions=True)

    if attentions is None:
        raise RuntimeError("Модель не вернула attention-карты.")

    matrix = prepare_attention_matrix(attentions, layer_idx=layer_idx, head_idx=head_idx)
    chars = list(tokenizer.decode(encoded))
    return chars, matrix
