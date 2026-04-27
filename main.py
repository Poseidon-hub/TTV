from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REQUIRED_CHECKPOINTS = ("model.pt", "vocab.json", "config.json")


def checkpoints_ready() -> bool:
    return all((CHECKPOINT_DIR / name).exists() for name in REQUIRED_CHECKPOINTS)


def run_streamlit() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Установите зависимости: pip install -r requirements.txt")
        return

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.showEmailPrompt",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("Запускаю Streamlit-интерфейс...")

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    try:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, env=env)
    except subprocess.CalledProcessError:
        print("Не удалось автоматически запустить Streamlit.")
        print(f"Попробуйте вручную: {' '.join(command)}")
    except Exception as exc:  # pragma: no cover - защитный fallback
        print(f"Ошибка запуска Streamlit: {exc}")
        print(f"Попробуйте вручную: {' '.join(command)}")


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if checkpoints_ready():
        print("Checkpoint-файлы найдены. Повторное обучение не требуется.")
    else:
        print("Checkpoint-файлы не найдены. Запускаю короткое обучение демо-модели...")
        train_model()

    if os.environ.get("TINY_TRANSFORMER_SKIP_STREAMLIT") == "1":
        print("TINY_TRANSFORMER_SKIP_STREAMLIT=1, запуск Streamlit пропущен.")
        return

    run_streamlit()


if __name__ == "__main__":
    main()
