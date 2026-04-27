from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.attention_utils import prepare_attention_from_step
from src.generate import generate_text_with_attention_steps, load_model_and_tokenizer
from src.visualization import build_attention_heatmap


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REQUIRED_FILES = ("model.pt", "vocab.json", "config.json")


def checkpoints_ready() -> bool:
    return all((CHECKPOINT_DIR / file_name).exists() for file_name in REQUIRED_FILES)


def display_char(ch: str) -> str:
    if ch == " ":
        return "[sp]"
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    return ch


def step_label(step_data: dict) -> str:
    step = int(step_data.get("step", 0))
    if step == 0:
        return "0 — исходная фраза"

    generated_char = display_char(str(step_data.get("generated_char", "")))
    return f"{step} — после генерации: '{generated_char}'"


st.set_page_config(
    page_title="TinyTransformer Visualizer",
    page_icon="🧠",
    layout="wide",
)

st.title("TinyTransformer Visualizer")
st.write("Russian Character-Level Transformer with Attention Visualization")
st.write(
    "Введите русскую фразу, сгенерируйте продолжение и посмотрите, "
    "как карта внимания меняется после каждого нового символа."
)
st.caption(
    "Attention пересчитывается после каждого нового символа. "
    "Выберите шаг генерации, чтобы увидеть, как менялась карта внимания модели."
)

if not checkpoints_ready():
    st.warning("Модель ещё не обучена. Запустите main.py или выполните обучение.")
    st.stop()

_, _, config, _ = load_model_and_tokenizer()

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""
if "attention_steps" not in st.session_state:
    st.session_state.attention_steps = []
if "selected_step_idx" not in st.session_state:
    st.session_state.selected_step_idx = 0

prompt = st.text_area("Введите русскую фразу", value="искусственный интеллект", height=100)
max_new_tokens = st.slider("max_new_tokens", min_value=20, max_value=300, value=120, step=10)
temperature = st.slider("temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.05)
layer_idx = st.selectbox("Слой attention", list(range(config.n_layer)))
head_idx = st.selectbox("Голова attention", list(range(config.n_head)))

if st.button("Сгенерировать", type="primary"):
    with st.spinner("Генерация текста и сбор attention по шагам..."):
        generated_text, attention_steps = generate_text_with_attention_steps(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    st.session_state.generated_text = generated_text
    st.session_state.attention_steps = attention_steps
    st.session_state.selected_step_idx = 0

if st.session_state.generated_text:
    st.subheader("Сгенерированный текст")
    st.write(st.session_state.generated_text)

steps = st.session_state.attention_steps
if steps:
    step_indices = list(range(len(steps)))
    if st.session_state.selected_step_idx not in step_indices:
        st.session_state.selected_step_idx = step_indices[-1]

    selected_index = st.selectbox(
        "Шаг генерации для attention",
        step_indices,
        format_func=lambda idx: step_label(steps[idx]),
        key="selected_step_idx",
    )

    selected_step = steps[selected_index]
    generated_char = str(selected_step.get("generated_char", ""))
    full_text = str(selected_step.get("full_text", selected_step.get("text", "")))
    context_text = str(selected_step.get("text", ""))

    st.write("Текущий текст на выбранном шаге:")
    st.code(full_text)

    if int(selected_step.get("step", 0)) == 0:
        st.write("Сгенерированный символ на этом шаге: (нет, это исходная фраза)")
    else:
        st.write(f"Сгенерированный символ на этом шаге: '{display_char(generated_char)}'")

    if full_text != context_text:
        st.caption(
            "Для расчёта attention модель использует последние "
            f"{config.block_size} символов (context window)."
        )

    chars, matrix = prepare_attention_from_step(selected_step, layer_idx=layer_idx, head_idx=head_idx)
    fig = build_attention_heatmap(
        chars,
        matrix,
        step=int(selected_step.get("step", 0)),
        generated_char=generated_char,
    )

    st.subheader("Attention heatmap")
    st.pyplot(fig, clear_figure=True)
