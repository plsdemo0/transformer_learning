# План реализации: Изучение трансформеров в Jupyter Notebook с PyTorch

Ветка: нет (git не настроен)
Создан: 2026-05-30

## Настройки
- Тестирование: нет
- Логирование: подробное (DEBUG)
- Документация: да (обязательная проверка)
- Режим: полный, без ветки

## Связь с Roadmap
Веха: Все вехи Roadmap
Описание: Полная реализация 15 вех по изучению архитектуры трансформеров

## План коммитов
> Информационно — git не настроен в проекте. Коммиты для ориентира при инициализации репозитория.
- **Коммит 1** (задачи 1–3): "feat: setup environment and tensor fundamentals"
- **Коммит 2** (задачи 4–5): "feat: linear layers, activations, gradients"
- **Коммит 3** (задачи 6–7): "feat: attention mechanism from scratch"
- **Коммит 4** (задачи 8–9): "feat: positional encodings and transformer encoder"
- **Коммит 5** (задачи 10–11): "feat: transformer decoder and full seq2seq"
- **Коммит 6** (задачи 12–13): "feat: training loop and BERT-style MLM"
- **Коммит 7** (задачи 14–15): "feat: GPT-style generation and visualization"
- **Коммит 8** (задачи 16–17): "feat: HuggingFace integration and final notebook"
- **Коммит 9** (задачи 18–19): "docs: README and requirements.txt"

## Задачи

### Фаза 1: Настройка и основы

- [x] **Задача 1: Настройка окружения (веха №1)**
  Установка PyTorch и создание базового ноутбука.
  - Проверить доступность ускорителей: CUDA (NVIDIA), MPS (Apple Silicon), CPU fallback
  - Создать `notebooks/01_environment_check.ipynb` с проверкой окружения
  - Установить зависимости: `torch`, `torchvision`, `torchtext`, `transformers`, `datasets`
  - Настроить `.venv` или обновить `requirements.txt`
  - Добавить логирование: `LOG_LEVEL` для всех экспериментов
  - Проверить, что `import torch` работает и тензоры создаются на GPU/MPS если доступно

  Файлы: `notebooks/01_environment_check.ipynb`, `requirements.txt`
  Логирование: DEBUG — версии библиотек, тип устройства, тестовые тензоры

- [x] **Задача 2: Основы тензоров PyTorch (веха №2)**
  Создать ноутбук для изучения тензоров и автограда.
  - Создание тензоров: из списков, numpy, random, zeros/ones/eye
  - Типы данных (dtype), устройства (device), форма (shape)
  - Индексация, срезы, решейп (view/reshape), транспонирование
  - Математические операции: сложение, умножение, матричное умножение
  - Broadcasting в PyTorch
  - Автоматическое дифференцирование: `requires_grad`, `backward()`, `grad`
  - Вычисление градиентов для простых функций
  - Визуализация графов вычислений (опционально через torchviz)

  Файлы: `notebooks/02_pytorch_tensors.ipynb`
  Логирование: DEBUG — формы тензоров, устройство, значения градиентов, контрольные точки

- [x] **Задача 3: Автоград и градиенты (продолжение вехи №2)**
  Углублённое изучение autograd.
  - `torch.no_grad()` и его применение
  - Остановка градиентов: `.detach()`
  - Пользовательские функции autograd (наследование `torch.autograd.Function`)
  - Градиенты нелинейных функций
  - Визуализация градиентов на простых моделях
  - Анализ Vanishing/Exploding gradients на синтетических данных

  Файлы: `notebooks/02_pytorch_tensors.ipynb` (продолжение)
  Логирование: DEBUG — значения градиентов на каждом шаге, норма градиентов

- [x] **Задача 4: Линейные слои и функции активации (веха №3)**
  Реализация полносвязных слоёв и активаций.
  - `nn.Linear` — устройство, весовая матрица, bias
  - Функции активации: ReLU, GELU (геллу), Sigmoid, Tanh — графики и производные
  - Почему GELU используется в трансформерах (сравнение с ReLU)
  - Построение простого классификатора на синтетических данных (circles/moons)
  - Функция потерь: CrossEntropyLoss, BCELoss
  - Визуализация градиентов в линейных слоях
  - Анализ: как меняются веса в процессе обучения

  Файлы: `notebooks/03_linear_layers_and_activations.ipynb`
  Логирование: DEBUG — веса до/после шага, градиенты, значения loss, точность

- [x] **Задача 5: Визуализация обучения (продолжение вехи №3)**
  Инструменты для мониторинга обучения.
  - Визуализация loss curves
  - Границы решений классификатора
  - Анимация обучения (matplotlib animation)
  - TensorBoard в ноутбуке: `%load_ext tensorboard`
  - Сравнение: разные LR, разные активации

  Файлы: `notebooks/03_linear_layers_and_activations.ipynb` (продолжение)
  Логирование: INFO — loss на каждой эпохе, DEBUG — детали градиентов

### Фаза 2: Механизмы внимания

- [x] **Задача 6: Scaled Dot-Product Attention с нуля (веха №4)**
  Реализация внимания с нуля на PyTorch.
  - Постановка задачи: почему нужно внимание? (зависимость от длины последовательности, контекст)
  - Query, Key, Value — интуиция и математика
  - Реализация: `Q @ K.T / sqrt(d_k)`, softmax, `@ V`
  - Маскировка: padding mask (для переменных длин)
  - Визуализация матрицы внимания (heatmap)
  - Демонстрация на синтетических данных (поиск соответствий в последовательностях)
  - Сравнение ручной реализации с `torch.nn.functional.scaled_dot_product_attention`

  Файлы: `notebooks/04_attention_from_scratch.ipynb`
  Логирование: DEBUG — формы Q, K, V, значения attention scores, паттерны маски

- [x] **Задача 7: Multi-Head Attention (веха №5)**
  Расширение до многоголового внимания.
  - Идея: разные головы учат разные паттерны взаимодействия
  - Реализация: разделение d_model на h голов по d_k = d_model/h
  - Проекционные слои: W_Q, W_K, W_V (общие для всех голов) → per-head split
  - Конкатенация голов и финальный проекционный слой W_O
  - Реализация класса `MultiHeadAttention(nn.Module)`
  - Анализ: что изучают разные головы? (визуализация)
  - Влияние числа голов на качество и скорость

  Файлы: `notebooks/05_multi_head_attention.ipynb`
  Логирование: DEBUG — число голов, размерность каждой головы, формы проекций

- [x] **Задача 8: Позиционные кодирования (веха №6)**
  Добавление информации о позиции токенов.
  - Проблема: attention инвариантен к порядку (permutation invariance)
  - Синусоидальные PE (Vaswani et al.): формулы, реализация, визуализация
    - Чередование sin/cos для разных частот
    - Визуализация: heatmap PE для разных позиций и размерностей
  - Обучаемые позиционные эмбеддинги (BERT-style): `nn.Embedding(max_len, d_model)`
  - Сравнение подходов: синусоидальные vs обучаемые
  - Относительные позиционные кодирования (краткий обзор: T5, ALiBi, RoPE)
  - Демонстрация: как PE влияет на внимание (сравнение с и без PE)

  Файлы: `notebooks/06_positional_encodings.ipynb`
  Логирование: DEBUG — значения PE для выбранных позиций, визуализация

### Фаза 3: Архитектура трансформера

- [x] **Задача 9: Transformer Encoder (веха №7)**
  Сборка энкодерного блока трансформера.
  - Компоненты: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
  - Layer Normalization: формула, реализация, зачем после attention
  - Остаточные связи (residual connections): почему они критичны
  - Feed-Forward Network: Linear → GELU → Linear (d_ff = 4 * d_model)
  - Dropout: применение в трансформерах
  - Реализация класса `TransformerEncoderBlock(nn.Module)`
  - Стекирование блоков: `TransformerEncoder(nn.Module)`
  - Визуализация: активации после каждого компонента
  - Тест: forward pass с синтетическими данными

  Файлы: `notebooks/07_transformer_encoder.ipynb`
  Логирование: DEBUG — формы на каждом шаге, значения attention до/после softmax, значения после LayerNorm

- [x] **Задача 10: Transformer Decoder (веха №8)**
  Реализация декодерного блока.
  - Masked Self-Attention: каузальная маскировка (не заглядывать в будущее)
  - Реализация каузальной маски: верхнетреугольная матрица с -inf
  - Cross-Attention: Q от декодера, K, V от энкодера
  - Реализация класса `TransformerDecoderBlock(nn.Module)`
  - Стекирование декодерных блоков: `TransformerDecoder(nn.Module)`
  - Отличие от энкодера: 3 под-слоя вместо 2
  - Визуализация: каузальная маска в действии

  Файлы: `notebooks/08_transformer_decoder.ipynb`
  Логирование: DEBUG — формы, каузальная маска, cross-attention scores

- [x] **Задача 11: Полный Transformer — Encoder-Decoder (веха №9)**
  Сборка полной seq2seq архитектуры.
  - Реализация `Transformer(nn.Module)`: encoder + decoder + output head
  - Токенизация: построение словаря, токенизация текста (char/word-level)
  - Эмбеддинги: `nn.Embedding(vocab_size, d_model)`
  - Output head: Linear → LogSoftmax (для генерации)
  - Демонстрация: задача копирования последовательностей
  - Генерация: авторегрессивная (по одному токену)
  - Визуализация: полная архитектура, поток данных

  Файлы: `notebooks/09_full_transformer.ipynb`
  Логирование: DEBUG — forward shapes, decoding step-by-step, generated tokens

- [x] **Задача 12: Обучение трансформера (веха №10)**
  Цикл обучения и оптимизация.
  - Функция потерь: CrossEntropyLoss с игнорированием padding
  - Оптимизатор: AdamW (рекомендация из LLM-практики)
  - Noam scheduler: увеличение LR (warmup) + экспоненциальное затухание
  - Реализация: `NoamScheduler` или `LambdaLR`
  - Цикл обучения с мониторингом: train loss, validation loss, accuracy
  - Ранняя остановка (early stopping)
  - Сохранение и загрузка модели: `torch.save`/`torch.load`
  - Эксперименты: влияние LR, warmup steps, d_model, num_heads
  - Визуализация: loss curves, LR schedule, сходимость

  Файлы: `notebooks/10_training_transformer.ipynb`
  Логирование: INFO — loss, accuracy, LR на каждой эпохе; DEBUG — градиенты, веса

### Фаза 4: Продвинутые архитектуры

- [x] **Задача 13: BERT — Masked Language Model (веха №11)**
  Реализация BERT-подобной модели (только энкодер).
  - Специальные токены: [CLS], [SEP], [MASK], [PAD]
  - BERT-архитектура: стопка энкодеров, CLS для классификации
  - Masked Language Model: маскировка 15% токенов
  - Реализация MLM head: Linear + Softmax над словарём
  - Демонстрация: предсказание замаскированных токенов
  - Контекстуальные представления: анализ эмбеддингов разных слоёв
  - Визуализация: t-SNE эмбеддингов, внимание в BERT

  Файлы: `notebooks/11_bert_mlm.ipynb`
  Логирование: DEBUG — маскированные позиции, предсказания, attention patterns

- [x] **Задача 14: GPT — Causal Language Model (веха №12)**
  Реализация GPT-подобной модели (только декодер).
  - GPT-архитектура: стопка декодеров (без cross-attention)
  - Авторегрессивная генерация: генерация по одному токену
  - Техники сэмплирования:
    - Greedy decoding (argmax)
    - Top-k sampling (выбор из k наиболее вероятных)
    - Top-p (nucleus) sampling (выбор из кумулятивной вероятности p)
    - Temperature scaling
  - Реализация генератора с различными стратегиями
  - Демонстрация: генерация текста

  Файлы: `notebooks/12_gpt_generation.ipynb`
  Логирование: DEBUG — вероятности токенов, top-k/top-p фильтры, сэмплированные токены

- [x] **Задача 15: Визуализация внимания (веха №13)**
  Извлечение и визуализация карт внимания.
  - Перехват attention weights из слоёв (hook)
  - Визуализация attention head-by-head (heatmaps)
  - Анализ: паттерны внимания на разных слоях
  - Сравнение: self-attention энкодера vs masked self-attention декодера vs cross-attention
  - Attention rollout: агрегация внимания через слои
  - Визуализация графа внимания (networkx или аналоги)
  - Интерактивная визуализация (ipywidgets)

  Файлы: `notebooks/13_attention_visualization.ipynb`
  Логирование: DEBUG — размерности attention weights, top-k вниманий для каждого токена

### Фаза 5: Интеграция и финализация

- [x] **Задача 16: HuggingFace Transformers (веха №14)**
  Работа с предобученными моделями.
  - Загрузка BERT: `AutoModel.from_pretrained('bert-base-uncased')`
  - Токенизация: `AutoTokenizer.from_pretrained('bert-base-uncased')`
  - Fine-tuning BERT для классификации текста (IMDb или AG News)
  - Загрузка GPT-2: генерация с предобученной моделью
  - Сравнение: наша реализация с нуля vs HuggingFace
  - Pipeline: `pipeline('text-classification')`, `pipeline('text-generation')`
  - Анализ: что предобученные модели "знают" о языке

  Файлы: `notebooks/14_huggingface_transformers.ipynb`
  Логирование: INFO — метрики fine-tuning, генерация примеров; DEBUG — токенизация

- [x] **Задача 17: Финальный ноутбук (веха №15)**
  Объединение всех концепций в один сквозной пример.
  - Цель: классификация текста (sentiment analysis) трансформером с нуля
  - Загрузка и подготовка датасета
  - Токенизация и построение словаря
  - Инициализация и обучение трансформера на small subset данных
  - Оценка: accuracy, F1, confusion matrix
  - Сравнение с предобученной моделью (HuggingFace BERT)
  - Выводы: что мы изучили, результаты, ограничения
  - Чистый, хорошо документированный код с комментариями

  Файлы: `notebooks/15_final_end_to_end.ipynb`
  Логирование: INFO — метрики, сравнение; DEBUG — детали обучения

- [x] **Задача 18: Обновление README и requirements.txt**
  Документирование проекта.
  - Обновить `README.md`: описание проекта, структура ноутбуков, инструкция по запуску
  - Обновить `requirements.txt`: все установленные зависимости с версиями
  - Проверить, что ноутбуки можно запустить последовательно

  Файлы: `README.md`, `requirements.txt`
  Логирование: не требуется

- [x] **Задача 19: Проверка документации (docs checkpoint)**
  Обязательная проверка документации.
  - Проверить актуальность README
  - Убедиться, что каждый ноутбук содержит:
    - Заголовок с описанием темы
    - Цели обучения
    - Импорты в начале
    - Визуализации результатов
    - Выводы в конце
  - Запустить `jupyter nbconvert --to script` для проверки отсутствия ошибок

  Файлы: все ноутбуки, README.md
  Логирование: INFO — результаты проверки
