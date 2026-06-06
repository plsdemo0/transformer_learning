#!/usr/bin/env python3
"""Накладывает текст на реальные мем-шаблоны из Imgflip."""

from PIL import Image, ImageDraw, ImageFont
import os

MEMES_DIR = os.path.join(os.path.dirname(__file__), "notebooks", "memes")
W, H = 800, 600

def get_font(size):
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_outline_text(draw, text, x, y, font, max_w=760):
    """Рисует текст с чёрным контуром. Переносит по словам при переполнении."""
    lines = []
    for word in text.split():
        if not lines:
            lines.append(word)
            continue
        test = lines[-1] + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_w:
            lines.append(word)
        else:
            lines[-1] = test

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lx = (W - lw) // 2
        for dx, dy in [(-2,-2), (-2,2), (2,-2), (2,2), (0,-2), (0,2), (-2,0), (2,0)]:
            draw.text((lx+dx, y+dy), line, font=font, fill="#000000")
        draw.text((lx, y), line, font=font, fill="#ffffff")
        y += bbox[3] - bbox[1] + 6


def meme_from_template(out_name, template_path, top_text, bottom_text, font_size=36):
    """Загружает шаблон, изменяет размер до 800x600, накладывает текст."""
    img = Image.open(template_path).convert("RGB")
    img = img.resize((W, H), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    draw_outline_text(draw, top_text, 0, 15, font)
    # Для нижнего текста меряем высоту
    dummy = ImageDraw.Draw(Image.new("RGB", (1,1)))
    bbox = dummy.textbbox((0, 0), bottom_text, font=font)
    line_h = bbox[3] - bbox[1] + 6
    # Прикидываем число строк
    num_lines = max(1, len(bottom_text) // 30 + 1)
    bottom_y = H - 20 - line_h * num_lines
    draw_outline_text(draw, bottom_text, 0, bottom_y, font)
    out_path = os.path.join(MEMES_DIR, out_name)
    img.save(out_path, quality=92)
    print(f"  ✓ {out_name}")
    return out_path


def make_meme(out_name, template_file, top_text, bottom_text, font_size=36):
    path = os.path.join(MEMES_DIR, template_file)
    if not os.path.exists(path):
        print(f"  ✗ {template_file} не найден, пропускаем")
        return
    meme_from_template(out_name, path, top_text, bottom_text, font_size)


# ==============================================================
# Генерация всех 15 мемов
# ==============================================================

# 1. Это нормально (This Is Fine)
make_meme("01_env.png", "_thisisfine.jpg",
    "КОГДА ПРОВЕРЯЕШЬ ОКРУЖЕНИЕ",
    "CUDA: нет • MPS: есть • CPU: ура, работает",
    34)

# 2. Expanding Brain — тензоры
make_meme("02_tensors.png", "_expandingbrain.jpg",
    "ТЕНЗОРНЫЕ РАЗМЕРНОСТИ",
    "0D • 1D • 2D • 3D • БАЧ, А ЧТО ЭТО ЗА 4-Я ОСЬ?!",
    32)

# 3. One Does Not Simply — линейные слои
make_meme("03_linear.png", "_simply.jpg",
    "ОДНИМ СЛОВОМ НЕ ОПИШЕШЬ",
    "ПОЧЕМУ ReLU НЕ ПРОПУСКАЕТ ОТРИЦАТЕЛЬНЫЕ?",
    36)

# 4. Drake — Attention vs RNN
make_meme("04_attention.png", "_drake.jpg",
    "RNN И ЭНКОДЕРЫ",
    "SELF-ATTENTION СО СКАЛИРОВАНИЕМ",
    34)

# 5. Two Buttons — Multi-Head
make_meme("05_mha.png", "_twobuttons.jpg",
    "8 ГОЛОВ ИЛИ 1 ГОЛОВА?",
    "...БЕРУ 8, БОЛЬШЕ ВЕСЕЛЕЕ",
    38)

# 6. Ancient Aliens — позиционные кодирования
make_meme("06_posenc.png", "_ancientaliens.jpg",
    "ТРАНСФОРМЕР БЕЗ ПОЗИЦИОННЫХ КОДИРОВАНИЙ",
    "...ТО ЕСТЬ МЫ НЕ ЗНАЕМ, КТО ЗА КЕМ СТОИТ?",
    32)

# 7. Change My Mind — LayerNorm
make_meme("07_encoder.png", "_changemymind.jpg",
    "BATCHNORM ЛУЧШЕ, ЧЕМ LAYERNORM",
    "...В ТРАНСФОРМЕРАХ",
    34)

# 8. Monkey Puppet — causal mask
make_meme("08_decoder.png", "_monkey.jpg",
    "БУДУЩИЕ ТОКЕНЫ:",
    "CAUSAL MASK: ✋ НИ-НИ-НИ",
    36)

# 9. Gru's Plan — полный трансформер
make_meme("09_full.png", "_grusplan.jpg",
    "ФАЗА 1: ЭНКОДЕР",
    "ФАЗА 2: ДЕКОДЕР • ФАЗА 3: ПРОФИТ",
    36)

# 10. Balloon — обучение
make_meme("10_training.png", "_balloon.jpg",
    "МОЙ LOSS В НАЧАЛЕ ОБУЧЕНИЯ",
    "МОЙ LOSS ПОСЛЕ 50 ЭПОХ",
    34)

# 11. Woman Yelling At Cat — BERT vs GPT
make_meme("11_bert.png", "_woman_cat.jpg",
    "BERT: ЗАПОЛНИ ПРОПУСК!",
    "[MASK] • [MASK] • [MASK] • Я ПРОСТО ХОТЕЛ СГЕНЕРИРОВАТЬ ТЕКСТ",
    32)

# 12. Always Has Been — GPT
make_meme("12_gpt.png", "_alwayshasbeen.jpg",
    "GPT-ГЕНЕРАЦИЯ:",
    "УВЕРЕННОСТЬ • ФАКТЫ • ИСТОЧНИКИ",
    36)

# 13. X Everywhere — визуализация внимания
make_meme("13_vis.png", "_xeverywhere.jpg",
    "ATTENTION HEATMAP:",
    "SUBJ • VERB • OBJ • [SEP] — ВСЕ ВНИМАНИЕ НА ВСЁ",
    34)

# 14. Trade Offer — HuggingFace
make_meme("14_hf.png", "_tradeoffer.jpg",
    "Я ПОЛУЧАЮ: ГОТОВУЮ МОДЕЛЬ",
    "ТЫ ПОЛУЧАЕШЬ: 3 СТРОЧКИ КОДА",
    36)

# 15. Buff Doge vs Cheems — финал
make_meme("15_final.png", "_buffdog.jpg",
    "Я ДО КУРСА",
    "Я ПОСЛЕ 15 НОУТБУКОВ О ТРАНСФОРМЕРАХ",
    32)

print(f"\n✓ Все 15 реальных мемов сохранены в {MEMES_DIR}")
