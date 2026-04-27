from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.config import ProjectConfig
from src.model import TinyTransformer
from src.tokenizer import CharTokenizer
from src.utils import get_device, get_project_root


PROJECT_ROOT = get_project_root()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"


def _check_required_files() -> None:
    required = [MODEL_PATH, VOCAB_PATH, CONFIG_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Отсутствуют checkpoint-файлы: {', '.join(missing)}")


def _load_model_and_tokenizer_uncached() -> Tuple[TinyTransformer, CharTokenizer, ProjectConfig, torch.device]:
    _check_required_files()
    tokenizer = CharTokenizer.load(VOCAB_PATH)
    config = ProjectConfig.load_json(CONFIG_PATH)
    if config.vocab_size == 0:
        config.vocab_size = tokenizer.vocab_size

    device = get_device()
    model = TinyTransformer(config).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, config, device


@lru_cache(maxsize=1)
def _cached_loader() -> Tuple[TinyTransformer, CharTokenizer, ProjectConfig, torch.device]:
    return _load_model_and_tokenizer_uncached()


def load_model_and_tokenizer(force_reload: bool = False) -> Tuple[TinyTransformer, CharTokenizer, ProjectConfig, torch.device]:
    if force_reload:
        _cached_loader.cache_clear()
    return _cached_loader()


def generate_text(prompt: str, max_new_tokens: int = 200, temperature: float = 0.8) -> str:
    model, tokenizer, _, device = load_model_and_tokenizer()

    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        fallback_id = tokenizer.stoi.get(" ", 0)
        encoded_prompt = [fallback_id]

    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    return tokenizer.decode(generated_ids[0].tolist())


def generate_text_with_attention_steps(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Генерирует текст посимвольно и сохраняет attention-карты для каждого шага.
    """

    model, tokenizer, config, device = load_model_and_tokenizer()

    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        encoded_prompt = [tokenizer.stoi.get(" ", 0)]

    max_new_tokens = max(0, int(max_new_tokens))
    temperature = max(float(temperature), 1e-5)

    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    attention_steps: List[Dict[str, Any]] = []
    generated_char_for_step = ""

    with torch.no_grad():
        for step in range(max_new_tokens + 1):
            idx_cond = idx[:, -config.block_size :]
            logits, _, attentions = model(idx_cond, return_attentions=True)

            if attentions is None:
                attentions = []

            step_attentions = [layer_attn.detach().cpu() for layer_attn in attentions]
            context_text = tokenizer.decode(idx_cond[0].tolist())
            full_text = tokenizer.decode(idx[0].tolist())

            attention_steps.append(
                {
                    "step": step,
                    "generated_char": generated_char_for_step,
                    "text": context_text,
                    "full_text": full_text,
                    "attentions": step_attentions,
                }
            )

            if step == max_new_tokens:
                break

            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            generated_char_for_step = tokenizer.decode([int(idx_next.item())])

    full_generated_text = tokenizer.decode(idx[0].tolist())
    return full_generated_text, attention_steps
