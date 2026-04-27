# TinyTransformer Visualizer: Russian Character-Level Transformer with Attention Visualization

## Краткое описание
Это небольшой образовательный ML-проект на русском языке.  
В проекте реализована компактная Transformer-модель, которая обучается продолжать русский текст на уровне символов и показывает attention-карты.

## Цель проекта
Показать, как работает Transformer изнутри, без использования готовых высокоуровневых реализаций.  
Код написан прозрачно: в `src/model.py` явно реализованы Q, K, V, causal mask и attention weights.

## Что показывает attention visualization
Attention heatmap показывает, на какие символы входной строки модель опирается при расчете контекста для предсказания следующего символа.

## Как устроен проект
- `main.py`:
  - проверяет наличие `checkpoints/model.pt`, `checkpoints/vocab.json`, `checkpoints/config.json`;
  - при отсутствии checkpoint автоматически запускает демо-обучение;
  - затем запускает Streamlit-интерфейс.
- `src/tokenizer.py`: character-level токенизатор.
- `src/model.py`: компактная Transformer-модель.
- `src/train.py`: обучение и сохранение checkpoint-файлов.
- `src/generate.py`: загрузка модели и генерация русского текста.
- `src/attention_utils.py`: извлечение attention-карт по слою/голове.
- `src/visualization.py`: построение heatmap через matplotlib.
- `app.py`: Streamlit-интерфейс для генерации и визуализации attention.

## Технологии
- Python
- PyTorch
- Streamlit
- Matplotlib
- NumPy
- HuggingFace Datasets (дополнительный программный источник русского текста, с fallback на локальный корпус)

## Запуск в PyCharm
1. Откройте файл `main.py`.
2. Нажмите `Shift + F10`.
3. При первом запуске модель автоматически обучится и сохранит checkpoint-файлы.
4. После обучения откроется локальный Streamlit-интерфейс.

Если зависимости не установлены:

```bash
pip install -r requirements.txt
```

## Attention по шагам генерации
- `step 0` показывает attention для исходной фразы до генерации.
- `step 1` показывает attention после первого сгенерированного символа.
- `step 2` показывает attention после второго символа, и так далее.
- На каждом новом шаге карта пересчитывается заново для текущего текста, поэтому можно увидеть, как меняется внимание модели в процессе генерации.
