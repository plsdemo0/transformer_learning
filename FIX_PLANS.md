# Планы исправления ноутбуков

Приоритет выполнения (сверху вниз):

---

## ~~1. Notebook 04 — `attention_from_scratch` (SyntaxError в benchmark)~~ **ГОТОВО**

~~- Заменить `\'cuda\'` → `'cuda'` и `\'mps\'` → `'mps'` (3 вхождения)~~

---

## ~~2. Notebook 05 — `multi_head_attention` (SyntaxError в benchmark)~~ **ГОТОВО**

~~- Заменить `\'cuda\'` → `'cuda'` и `\'mps\'` → `'mps'` (3 вхождения)~~

---

## ~~4. Notebook 09 — `full_transformer` (генерация сломана)~~ **ГОТОВО**

~~- Увеличить `max_len` в `model_small`: 8 → 50~~

---

## ~~5. Notebook 10 — `training_transformer` (ValueError batch size)~~ **ГОТОВО**

~~- Исправить `make_data()` — добавить `[BOS]` в tgt~~

---

## ~~6. Notebook 11 — `bert_mlm` (device mismatch)~~ **ГОТОВО**

~~- Добавить `device=input_ids.device` ко всем `torch.full()` и `torch.randint()` в `mask_tokens()`~~

---

## ~~7. Notebook 12 — `gpt_generation` (device mismatch)~~ **ГОТОВО**

~~- Добавить `.to(device)` к `prompt`~~

---

## ~~8. Notebook 13 — `attention_visualization` (3x RuntimeError)~~ **ГОТОВО**

~~- Заменить `.numpy()` → `.detach().cpu().numpy()` (3 места)~~
~~- Добавить `.detach()` в `EncoderWithHooks.forward()`~~

---

## ~~9. Notebook 14 — `huggingface_transformers` (3 ошибки)~~ **ГОТОВО**

~~- Заменить `load_dataset("imdb")` → `"stanfordnlp/imdb"`~~
~~- Заменить `evaluation_strategy` → `eval_strategy`~~

---

## ~~10. Notebook 15 — `final_end_to_end` (каскадный HfUriError)~~ **ГОТОВО**

~~- Заменить `load_dataset("imdb")` → `"stanfordnlp/imdb"`~~
~~- Padding mask вычислять до embedding, по raw token IDs~~

---

## ~~11. Notebook 02 — `pytorch_tensors` (нет тренировочного цикла)~~ **ГОТОВО**

~~**План:**~~
~~- Добавить ячейку в конец: мини-линейная регрессия на синтетических данных y = 2x + 1 + noise~~
~~- Обучить SGD: создание данных → модель → loss → backward → update~~

---

## ~~12. Notebook 03 — `linear_layers_and_activations` (дополнения)~~ **ГОТОВО**

~~- Добавить evaluation на held-out test set с train/test split~~
~~- Добавить grid search по learning rate с визуализацией~~
