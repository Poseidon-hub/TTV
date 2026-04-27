from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.config import ProjectConfig
from src.model import TinyTransformer
from src.tokenizer import CharTokenizer
from src.utils import ensure_directory, get_device, get_project_root, set_seed


PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "russian_corpus.txt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"


def _load_external_worker(max_samples: int, output_queue: "mp.Queue[Dict[str, str]]") -> None:
    try:
        from datasets import load_dataset
    except Exception:
        output_queue.put({"text": "", "meta": "", "error": ""})
        return

    try:
        candidate_configs = ("en-ru", "de-ru", "fr-ru", "it-ru", "es-ru")
        for config_name in candidate_configs:
            try:
                dataset = load_dataset("opus_books", config_name, split=f"train[:{max_samples}]")
            except Exception:
                continue

            texts = []
            for item in dataset:
                ru_text = item.get("translation", {}).get("ru", "")
                if ru_text:
                    texts.append(ru_text.strip())

            loaded_text = "\n".join(texts)
            if loaded_text:
                output_queue.put(
                    {
                        "text": loaded_text,
                        "meta": (
                            "Загружен дополнительный корпус из HuggingFace datasets: "
                            f"{len(texts)} строк (opus_books/{config_name})."
                        ),
                        "error": "",
                    }
                )
                return

        output_queue.put({"text": "", "meta": "", "error": ""})
    except Exception as exc:  # pragma: no cover - защитный fallback
        output_queue.put({"text": "", "meta": "", "error": str(exc)})


def load_external_russian_text(max_samples: int = 200, timeout_seconds: int = 20) -> str:
    """
    Пытается программно загрузить дополнительный русскоязычный корпус из библиотеки datasets.
    """

    ctx = mp.get_context("spawn")
    output_queue: "mp.Queue[Dict[str, str]]" = ctx.Queue()
    process = ctx.Process(target=_load_external_worker, args=(max_samples, output_queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        print(
            "Загрузка внешнего датасета заняла слишком много времени "
            f"(>{timeout_seconds} сек). Продолжаю обучение на локальном корпусе."
        )
        return ""

    payload = {"text": "", "meta": "", "error": ""}
    if not output_queue.empty():
        payload = output_queue.get()

    if payload["error"]:
        print(f"Не удалось загрузить внешний датасет из библиотеки: {payload['error']}")
        return ""

    if payload["meta"]:
        print(payload["meta"])

    return payload["text"]


def read_training_text() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден корпус: {DATA_PATH}")

    local_text = DATA_PATH.read_text(encoding="utf-8").strip()
    external_text = load_external_russian_text()

    if external_text:
        return f"{local_text}\n{external_text}"
    return local_text


def make_batches(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.size(0) <= block_size + 1:
        # Если данных мало, повторяем их, чтобы можно было сделать batch.
        repeat_factor = (block_size + batch_size + 2) // max(int(data.size(0)), 1) + 1
        data = data.repeat(repeat_factor)

    max_start = data.size(0) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: Dict[str, float] = {}

    for split_name, split_data in (("train", train_data), ("val", val_data)):
        split_losses = []
        for _ in range(config.eval_iters):
            xb, yb = make_batches(split_data, config.block_size, config.batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss, _ = model(xb, yb)
            split_losses.append(float(loss.item()))
        losses[split_name] = sum(split_losses) / max(len(split_losses), 1)

    model.train()
    return losses


def train_model(config: ProjectConfig | None = None) -> None:
    config = config or ProjectConfig()
    set_seed(config.seed)
    ensure_directory(CHECKPOINT_DIR)

    text = read_training_text()
    if not text:
        raise ValueError("Корпус пустой. Добавьте текст в data/russian_corpus.txt.")

    tokenizer = CharTokenizer().build_from_text(text)
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("После токенизации слишком мало данных для обучения.")

    data = torch.tensor(ids, dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:] if split_idx < len(data) else data[:]
    if val_data.numel() == 0:
        val_data = train_data

    train_config = replace(config, vocab_size=tokenizer.vocab_size)
    device = get_device()
    model = TinyTransformer(train_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    print(f"Устройство: {device}")
    print(f"Размер словаря: {train_config.vocab_size} символов")
    print("Старт обучения...")

    for step in range(train_config.max_iters + 1):
        if step % train_config.eval_interval == 0 or step == train_config.max_iters:
            losses = estimate_loss(model, train_data, val_data, train_config, device)
            print(
                f"iter {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )

        xb, yb = make_batches(train_data, train_config.block_size, train_config.batch_size)
        xb = xb.to(device)
        yb = yb.to(device)

        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)
    tokenizer.save(VOCAB_PATH)
    train_config.save_json(CONFIG_PATH)

    print("Обучение завершено. Файлы сохранены:")
    print(f"- {MODEL_PATH}")
    print(f"- {VOCAB_PATH}")
    print(f"- {CONFIG_PATH}")
