import json

path = r'D:\OpenCodeNLLM\OpenCode\Transformer_Notebooks\transformer_learning\notebooks\14_huggingface_transformers.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

for i, cell in enumerate(cells):
    if cell['cell_type'] != 'code':
        continue

    new_lines = None

    if i == 1:  # imports
        new_lines = [
            "# Импорты: стандартная библиотека, PyTorch, NumPy, Matplotlib",
            "import sys, os, math",
            "import torch",
            "import torch.nn as nn",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "# Автоопределение устройства: CUDA > MPS > CPU",
            "if torch.cuda.is_available():",
            "    device = torch.device('cuda')",
            "elif torch.backends.mps.is_available():",
            "    device = torch.device('mps')",
            "else:",
            "    device = torch.device('cpu')",
            "",
        ]

    elif i == 2:  # huggingface imports
        new_lines = [
            "# Импорты HuggingFace: токенизатор, модели, пайплайны, Trainer",
            "from transformers import (",
            "    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,",
            "    pipeline, Trainer, TrainingArguments, BertForSequenceClassification",
            ")",
            "from datasets import load_dataset",
            "",
        ]

    elif i == 4:  # distilbert
        new_lines = [
            "# distilbert — компактная версия BERT (быстрее, чуть хуже точность)",
            "model_name = 'distilbert-base-uncased'",
            "tokenizer = AutoTokenizer.from_pretrained(model_name)",
            "bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)",
            "",
            "# Демонстрация токенизации: BERT использует WordPiece (подсловные токены)",
            "text = 'Transformers are amazing for natural language processing!'",
            "tokens = tokenizer(text, return_tensors='pt').to(device)",
            "print(f'Original: {text}')",
            "print(f'Tokens:   {tokenizer.convert_ids_to_tokens(tokens[\"input_ids\"][0])}')",
            "print(f'Input IDs: {tokens[\"input_ids\"][0].tolist()}')",
        ]

    elif i == 6:  # dataset
        new_lines = [
            "",
            "# Загружаем подмножество IMDB: 100 тренировочных, 50 тестовых",
            "dataset = load_dataset('stanfordnlp/imdb', split=['train[:100]', 'test[:50]'])",
            "train_dataset, test_dataset = dataset",
            "",
            "print(f'Train samples: {len(train_dataset)}')",
            "print(f'Test samples:  {len(test_dataset)}')",
            "print(f'Example: {train_dataset[0][\"text\"][:100]}...')",
            "print(f'Label: {train_dataset[0][\"label\"]} (0=neg, 1=pos)')",
            "",
        ]

    elif i == 7:  # tokenization function
        new_lines = [
            "# Функция токенизации: padding до макс. длины, обрезка длинных текстов",
            "def tokenize_function(examples):",
            "    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)",
            "",
            "# .map() применяет функцию ко всем примерам в датасете (batched=True — быстрее)",
            "train_enc = train_dataset.map(tokenize_function, batched=True)",
            "test_enc = test_dataset.map(tokenize_function, batched=True)",
            "",
            "# Trainer ожидает поле 'labels' (не 'label') — переименовываем",
            "train_enc = train_enc.rename_column('label', 'labels')",
            "test_enc = test_enc.rename_column('label', 'labels')",
            "# Переключаем формат на torch.Tensor для совместимости с Trainer",
            "train_enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])",
            "test_enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])",
            "",
        ]

    elif i == 8:  # training
        new_lines = [
            "# TrainingArguments — гиперпараметры обучения",
            "training_args = TrainingArguments(",
            "    output_dir='./checkpoints/hf-bert-imdb',",
            "    num_train_epochs=2,",
            "    per_device_train_batch_size=8,",
            "    per_device_eval_batch_size=8,",
            "    logging_steps=5,",
            "    eval_strategy='steps',  # оценка каждые eval_steps шагов",
            "    eval_steps=10,",
            "    save_strategy='no',  # не сохраняем чекпоинты",
            "    report_to='none',",
            "    disable_tqdm=True,",
            ")",
            "",
            "# Trainer — высокоуровневое API: инкапсулирует цикл обучения",
            "trainer = Trainer(",
            "    model=bert_model,",
            "    args=training_args,",
            "    train_dataset=train_enc,",
            "    eval_dataset=test_enc,",
            ")",
            "",
            "# Запуск: Trainer сам управляет forward/backward/optimizer",
            "trainer.train()",
            "",
        ]

    elif i == 9:  # evaluation
        new_lines = [
            "# Оценка на тестовой выборке — возвращает dict с метриками",
            "results = trainer.evaluate()",
            "print(f'Evaluation loss: {results[\"eval_loss\"]:.4f}')",
            "",
            "# Ручная проверка на нескольких примерах для интуиции",
            "test_texts = [",
            "    'This movie was absolutely fantastic! I loved every minute.',",
            "    'Terrible waste of time. The acting was horrible.',",
            "    'It was okay, not great but not terrible either.',  # нейтральный — сложный случай",
            "]",
            "for text in test_texts:",
            "    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)",
            "    with torch.no_grad():",
            "        outputs = bert_model(**inputs)",
            "    pred = outputs.logits.argmax(-1).item()  # индекс класса с макс. вероятностью",
            "    sentiment = 'positive' if pred == 1 else 'negative'",
            "    print(f'  \"{text[:50]}...\" -> {sentiment}')",
            "",
        ]

    elif i == 11:  # GPT-2
        new_lines = [
            "# Загружаем GPT-2 — авторегрессионная модель для генерации текста",
            "gpt2_name = 'gpt2'",
            "gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_name)",
            "gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device)",
            "gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # GPT-2 не имеет pad_token — используем eos",
            "",
            "prompt = 'The future of artificial intelligence is'",
            "inputs = gpt2_tokenizer(prompt, return_tensors='pt').to(device)",
            "",
            "# Генерация с top-p (nucleus) сэмплингом и температурой",
            "with torch.no_grad():",
            "    outputs = gpt2_model.generate(",
            "        **inputs,",
            "        max_new_tokens=50,",
            "        temperature=0.8,  # < 1 — более уверенные предсказания",
            "        top_p=0.9,  # nucleus: выбираем из токенов с кумулятивной вероятностью 90%",
            "        do_sample=True,  # стохастическая выборка (не greedy)",
            "        pad_token_id=gpt2_tokenizer.eos_token_id,",
            "    )",
            "",
            "generated = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)",
            "print(f'Prompt: {prompt}')",
            "print(f'Generated: {generated}')",
            "",
        ]

    elif i == 13:  # pipeline
        new_lines = [
            "# Pipeline API — универсальный интерфейс (токенизация + модель + постобработка)",
            "classifier = pipeline('text-classification', model=bert_model, tokenizer=tokenizer, device=0 if device.type in ('cuda', 'mps') else -1)",
            "",
            "samples = [",
            "    'I really enjoyed this film, great acting!',",
            "    'This was boring and way too long.',",
            "]",
            "# Pipeline автоматически батчит и возвращает список result[{'label', 'score'}]",
            "for text, result in zip(samples, classifier(samples)):",
            "    print(f'  \"{text[:40]}...\" -> {result[\"label\"]} (score: {result[\"score\"]:.3f})')",
            "",
            "# Тот же Pipeline для генерации текста — единый интерфейс",
            "generator = pipeline('text-generation', model=gpt2_model, tokenizer=gpt2_tokenizer, device=0 if device.type in ('cuda', 'mps') else -1)",
            "result = generator('Transformers have revolutionized', max_new_tokens=30, num_return_sequences=1)",
            "print(f'  Generated: {result[0][\"generated_text\"]}')",
            "",
        ]

    elif i == 15:  # summary
        new_lines = [
            "# Итоговый вывод с перечислением изученного в этом ноутбуке",
            'print("=== HuggingFace Transformers complete ===")',
            'print("Topics covered:")',
            'print("  - Loading BERT (distilbert) with AutoTokenizer/AutoModel")',
            'print("  - Fine-tuning BERT for text classification on IMDb")',
            'print("  - Trainer API")',
            'print("  - GPT-2 text generation (top-p, temperature)")',
            'print("  - HuggingFace Pipeline API")',
            'print("  - Comparison: our implementation vs HuggingFace")',
            "",
        ]

    if new_lines is not None:
        cell['source'] = [l + '\n' for l in new_lines]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook 14 done")
