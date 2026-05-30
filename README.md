# Ashat_AI — Изучение трансформеров

Проект для изучения архитектуры трансформеров с нуля на PyTorch — от Scaled Dot-Product Attention до полного Transformer и fine-tuning через HuggingFace.

## Структура ноутбуков

### Фаза 1: Основы
| # | Ноутбук | Тема |
|---|---------|------|
| 1 | `01_environment_check.ipynb` | Проверка окружения (CUDA/MPS/CPU) |
| 2 | `02_pytorch_tensors.ipynb` | Тензоры, автоград, градиенты |
| 3 | `03_linear_layers_and_activations.ipynb` | Linear, ReLU, GELU, классификатор |

### Фаза 2: Механизмы внимания
| # | Ноутбук | Тема |
|---|---------|------|
| 4 | `04_attention_from_scratch.ipynb` | Scaled Dot-Product Attention |
| 5 | `05_multi_head_attention.ipynb` | Multi-Head Attention |
| 6 | `06_positional_encodings.ipynb` | Синусоидальные/обучаемые PE, ALiBi |

### Фаза 3: Архитектура трансформера
| # | Ноутбук | Тема |
|---|---------|------|
| 7 | `07_transformer_encoder.ipynb` | Encoder (Self-Attention + FFN + Add&Norm) |
| 8 | `08_transformer_decoder.ipynb` | Decoder (Causal + Cross-Attention) |
| 9 | `09_full_transformer.ipynb` | Полный Encoder-Decoder + копирование |
| 10 | `10_training_transformer.ipynb` | Цикл обучения, Noam scheduler, early stopping |

### Фаза 4: Продвинутые архитектуры
| # | Ноутбук | Тема |
|---|---------|------|
| 11 | `11_bert_mlm.ipynb` | BERT — Masked Language Model |
| 12 | `12_gpt_generation.ipynb` | GPT — Causal LM, top-k, top-p, temperature |
| 13 | `13_attention_visualization.ipynb` | Визуализация внимания, attention rollout |

### Фаза 5: Интеграция
| # | Ноутбук | Тема |
|---|---------|------|
| 14 | `14_huggingface_transformers.ipynb` | HuggingFace: BERT fine-tuning, GPT-2 |
| 15 | `15_final_end_to_end.ipynb` | Сквозной пример + сравнение с BERT |

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Запуск

```bash
# Настройка уровня логирования
export LOG_LEVEL=INFO  # или DEBUG для подробных логов

# Запуск Jupyter
jupyter lab
# или
jupyter notebook
```

Ноутбуки рекомендуется запускать последовательно, так как каждый опирается на концепции из предыдущих.

## Технологии

- Python 3.9
- PyTorch 2.8+
- Transformers (HuggingFace)
- JupyterLab
- Matplotlib, NumPy, scikit-learn

## Результаты

Наш трансформер с нуля (200MB данных, char-level) ~55-65% accuracy на IMDb.
DistilBERT (предобученный) ~85-90% accuracy на том же подмножестве.
