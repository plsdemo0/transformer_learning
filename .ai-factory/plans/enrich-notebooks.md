# Углубление материалов ноутбуков

**Ветка:** — (git отключён)
**Дата:** 2026-05-31
**Тип:** enhancement

## Настройки

- **Тестирование:** нет
- **Логирование:** подробное (DEBUG)
- **Документация:** обязательный чекпоинт

## Roadmap Linkage

- **Веха:** Углубление материалов
- **Обоснование:** Все 15 ноутбуков по трансформерам реализованы; следующий раунд — enrich: добавление подробных комментариев, теории и ссылок

---

## Research Context

Из explore-сессии: в проекте 15 ноутбуков, 97 markdown-ячеек, 149 code-ячеек, 132 inline-комментария. Только 5/15 содержат LaTeX-формулы, только 5/15 — ссылки на статьи. Цель: довести каждый ноутбук до уровня "понятно новичку с нуля".

---

## Задачи

### Фаза 1: DESCRIPTION + Roadmap

#### 1.1 — Обновить DESCRIPTION.md

- [x] Сменить фокус с food-tech на обучение трансформерам
- [x] Описать назначение каждого из 15 ноутбуков
- [x] Указать технологии (PyTorch, HuggingFace, matplotlib)

Файл: `.ai-factory/DESCRIPTION.md`

#### 1.2 — Обновить ROADMAP.md: добавить веху "Углубление материалов"

- [x] Добавить новую веху `[ ] Углубление материалов — добавление комментариев, теории и ссылок во все 15 ноутбуков`
- [x] Дата начала: 2026-05-31

Файл: `.ai-factory/ROADMAP.md`

---

### Фаза 2: Enrich ноутбуков 01-03 (Основы)

#### 2.1 — 01_environment_check.ipynb

- [x] Добавить комментарии в каждую ячейку кода (что делает `torch.backends.mps.is_available()`, почему выбираем устройство)
- [x] Добавить markdown-ячейку "Почему важен device" (разница CPU/MPS/CUDA, transfer overhead)
- [x] Ссылка на документацию PyTorch по MPS

Файл: `notebooks/01_environment_check.ipynb`

#### 2.2 — 02_pytorch_tensors.ipynb

- [x] Добавить комментарии: объяснение broadcasting (правила), почему `view` vs `reshape`, что такое `contiguous`, как работает autograd (граф вычислений)
- [x] Добавить markdown-ячейки:
  - "Что такое broadcasting и как он работает" (с примерами)
  - "Autograd: как PyTorch строит граф вычислений"
  - "Размерности в нейросетях: batch, features, sequence"
- [x] Формулы: MSE loss в LaTeX

Файл: `notebooks/02_pytorch_tensors.ipynb`

#### 2.3 — 03_linear_layers_and_activations.ipynb

- [x] Добавить комментарии: что делает Linear (матричное умножение + bias), зачем GELU, для чего нужна инициализация весов
- [x] Добавить markdown-ячейки:
  - "Почему нелинейности критичны" (визуализация без активации = линейная комбинация)
  - "Инициализация весов: Xavier, Kaiming, зачем это"
- [x] Формулы: ReLU, GELU, Sigmoid в LaTeX

Файл: `notebooks/03_linear_layers_and_activations.ipynb`

---

### Фаза 3: Enrich ноутбуков 04-06 (Attention)

#### 3.1 — 04_attention_from_scratch.ipynb

- [x] Добавить комментарии к каждой операции: `Q @ K.T / sqrt(d_k)`, зачем делить на `sqrt(d_k)`, что делает softmax, как работает маскировка
- [x] Добавить markdown-ячейки:
  - "Почему Scaled Dot-Product: анализ дисперсии" (график распределения до/после scaling)
  - "Вывод градиентов через attention"
  - "Сложность O(n²·d): откуда берётся квадрат"
- [x] Визуализация: распределение scores до и после scaling

Файл: `notebooks/04_attention_from_scratch.ipynb`

#### 3.2 — 05_multi_head_attention.ipynb

- [x] Добавить комментарии: почему делим d_model на n_heads, что даёт多头, как работают `transpose` для входа и выхода
- [x] Добавить markdown-ячейки:
  - "Что дают разные головы: анализ на примере" (ссылка на визуализации из Transformer)
  - "Параметры: W_Q, W_K, W_V, W_O — линейные проекции"
- [x] Формулы: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

Файл: `notebooks/05_multi_head_attention.ipynb`

#### 3.3 — 06_positional_encodings.ipynb

- [x] Добавить комментарии: почему sin/cos разных частот, как PE складывается с эмбеддингом, почему PE сохраняет относительные позиции
- [x] Добавить markdown-ячейки:
  - "Вывод: почему sin/cos сохраняют относительное расстояние" (тригонометрические тождества)
  - "Сравнение: Sinusoidal vs Learnable vs RoPE vs ALiBi"
- [x] Формулы: PE(pos,2i) = sin(pos/10000^(2i/d))

Файл: `notebooks/06_positional_encodings.ipynb`

---

### Фаза 4: Enrich ноутбуков 07-09 (Encoder, Decoder, Full)

#### 4.1 — 07_transformer_encoder.ipynb

- [x] Добавить комментарии: Pre-Norm vs Post-Norm, зачем LayerNorm (не BatchNorm), роль residual connections
- [x] Добавить markdown-ячейки:
  - "Почему LayerNorm, а не BatchNorm" (анализ статистик, зависимость от batch)
  - "Residual connections: как решают затухание градиентов"
- [x] Ссылки: "Attention is All You Need" LayerNorm section, Pre-LN paper

Файл: `notebooks/07_transformer_encoder.ipynb`

#### 4.2 — 08_transformer_decoder.ipynb

- [x] Добавить комментарии: как работает causal mask (почему diagonal=1, почему `-inf`, а не `0`), cross-attention (Q от decoder, K/V от encoder)
- [x] Добавить markdown-ячейки:
  - "Causal masking: почему будущие токены обнуляются"
  - "Cross-attention: как encoder передаёт информацию decoder'у"

Файл: `notebooks/08_transformer_decoder.ipynb`

#### 4.3 — 09_full_transformer.ipynb

- [x] Добавить комментарии: как работает teacher forcing, итеративное декодирование, разница train vs inference
- [x] Добавить markdown-ячейки:
  - "Teacher forcing: обучение с учителем vs авторегрессия на инференсе"
  - "Копирование последовательностей: почему это тестовая задача"

Файл: `notebooks/09_full_transformer.ipynb`

---

### Фаза 5: Enrich ноутбуков 10-12 (Training, BERT, GPT)

#### 5.1 — 10_training_transformer.ipynb

- [x] Добавить комментарии: Noam scheduler (формула), early stopping, checkpointing
- [x] Добавить markdown-ячейки:
  - "Noam scheduler: разбор формулы, почему warmup + decay"
  - "AdamW vs Adam: зачем weight decay"
  - "Overfitting в трансформерах: dropout, weight decay, early stopping"
- [x] Formulas: Noam: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

Файл: `notebooks/10_training_transformer.ipynb`

#### 5.2 — 11_bert_mlm.ipynb

- [x] Добавить комментарии: 80/10/10 masking strategy, как работает MLM head, почему BERT — encoder-only
- [x] Добавить markdown-ячейки:
  - "BERT: почему encoder-only (а не decoder) для понимания"
  - "80/10/10 стратегия маскировки: почему не 100%"
- [x] Ссылки: BERT paper, MLM objective

Файл: `notebooks/11_bert_mlm.ipynb`

#### 5.3 — 12_gpt_generation.ipynb

- [x] Добавить комментарии: top-k vs top-p vs temperature — что и как работает, роль causal LM
- [x] Добавить markdown-ячейки:
  - "Top-p (nucleus) sampling: отсечение хвоста распределения"
  - "Temperature: размягчение распределения softmax"
  - "Greedy vs Sampling: trade-off качество vs разнообразие"
- [x] Формулы: softmax с temperature

Файл: `notebooks/12_gpt_generation.ipynb`

---

### Фаза 6: Enrich ноутбуков 13-15 (Visualization, HF, Final)

#### 6.1 — 13_attention_visualization.ipynb

- [x] Добавить комментарии: hooks для извлечения внимания, attention rollout
- [x] Добавить markdown-ячейки:
  - "Attention Rollout: как агрегировать внимание через слои"
- [x] Ссылки: Abnar & Zuidema 2020

Файл: `notebooks/13_attention_visualization.ipynb`

#### 6.2 — 14_huggingface_transformers.ipynb

- [x] Добавить комментарии: что делает HuggingFace Pipeline, разница Trainer API vs manual loop
- [x] Добавить markdown-ячейки:
  - "HuggingFace ecosystem: AutoModel, Trainer, Pipeline"
  - "Fine-tuning: transfer learning с предобученной модели"

Файл: `notebooks/14_huggingface_transformers.ipynb`

#### 6.3 — 15_final_end_to_end.ipynb

- [x] Добавить комментарии: char-level vs subword tokenization, использование BOS как CLS
- [x] Добавить markdown-ячейки:
  - "Char-level vs Word-level vs Subword токенизация"
  - "Почему BOS токен можно использовать как CLS"

Файл: `notebooks/15_final_end_to_end.ipynb`

---

## Commit Plan

Проект не использует git. Изменения сохраняются напрямую.
