# Ashat_AI — Описание проекта

## Обзор
Проект "Ashat_AI" — учебный курс по архитектуре трансформеров с практической реализацией на PyTorch. 15 Jupyter Notebook, от основ тензоров до полного трансформера и сравнения с HuggingFace.

## Структура ноутбуков

| # | Тема |
|---|------|
| 01 | Проверка окружения, device (CPU/MPS/CUDA) |
| 02 | Основы тензоров PyTorch, broadcasting, autograd |
| 03 | Линейные слои, функции активации, простой классификатор |
| 04 | Scaled Dot-Product Attention (с нуля) |
| 05 | Multi-Head Attention |
| 06 | Позиционные кодирования (sinusoidal, learnable, ALiBi) |
| 07 | Transformer Encoder (LayerNorm, FeedForward, Pre-Norm) |
| 08 | Transformer Decoder (causal mask, cross-attention) |
| 09 | Полный Transformer (Encoder-Decoder, seq2seq) |
| 10 | Обучение трансформера (Noam scheduler, early stopping) |
| 11 | BERT: Masked Language Modelling |
| 12 | GPT: Causal Language Modelling, стратегии сэмплирования |
| 13 | Визуализация внимания (hooks, attention rollout) |
| 14 | HuggingFace Transformers (fine-tuning, pipeline) |
| 15 | Финальный сквозной пример: классификация текста |

## Технологический стек
- **Язык:** Python 3.9.6
- **Библиотеки:** torch, numpy, matplotlib, transformers, datasets
- **Виртуальное окружение:** .venv/
- **Формат:** Jupyter Notebook

## Архитектурные заметки
- Каждый ноутбук самодостаточен (переиспользует код из предыдущих)
- Логирование через модуль `logging` с конфигурацией через переменную окружения `LOG_LEVEL`
- Данные: `data/`, модели: `models/`, зависимости: `requirements.txt`

## Требования
- Настройка логирования через переменные окружения
- Обработка ошибок с явным логированием
