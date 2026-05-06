import asyncio
import logging
import os
import io
from typing import List, Tuple
import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

DEFAULT_N_COLORS = 24
MIN_REGION_SIZE = 150
MAX_IMAGE_SIZE = 1500

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Предобработка изображения"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Лёгкое сглаживание для уменьшения шума
    img_array = cv2.bilateralFilter(img_array, 7, 50, 50)
    
    return img_array


def cluster_colors(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Квантование цветов"""
    h, w = img_array.shape[:2]
    pixels = img_array.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    
    # Сортировка по яркости
    brightness = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    sorted_indices = np.argsort(brightness)
    centers = centers[sorted_indices]
    
    # Переназначаем метки
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels = np.array([label_map[l] for l in labels]).reshape((h, w))
    
    quantized = centers[labels]
    
    return quantized, labels, [tuple(c) for c in centers]


def find_connected_regions(mask: np.ndarray, min_size: int = 100) -> List[Tuple[List[int], List[int], int, int]]:
    """
    Поиск связных областей через BFS (как в оригинальном IBNG)
    Возвращает список регионов: (x_координаты, y_координаты, центр_x, центр_y)
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                # BFS
                queue = deque([(x, y)])
                region_x, region_y = [], []
                
                while queue:
                    cx, cy = queue.popleft()
                    if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] and not visited[cy, cx]:
                        visited[cy, cx] = True
                        region_x.append(cx)
                        region_y.append(cy)
                        queue.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                
                if len(region_x) >= min_size:
                    center_x = int(np.mean(region_x))
                    center_y = int(np.mean(region_y))
                    regions.append((region_x, region_y, center_x, center_y))
    
    return regions


def get_center_in_region(mask: np.ndarray) -> Tuple[int, int]:
    """
    Умный поиск центра области (адаптировано из IBNG)
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0
    
    # Ищем центр по вертикали и горизонтали
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    
    # Корректируем, чтобы центр был внутри
    if not mask[center_y, center_x]:
        # Ищем ближайшую точку внутри
        distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
        idx = np.argmin(distances)
        center_x, center_y = xs[idx], ys[idx]
    
    return center_x, center_y


def create_coloring_page(img_array: np.ndarray, quantized: np.ndarray, labels: np.ndarray, 
                        palette: List[Tuple[int, int, int]], min_region_size: int) -> Image.Image:
    """
    Создание раскраски по номерам (вдохновлено IBNG)
    """
    h, w = img_array.shape[:2]
    
    # Создаём холст
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Сначала находим ВСЕ контуры
    all_contours_mask = np.zeros((h, w), dtype=bool)
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2)
        
        # Находим контуры через разницу (метод из IBNG)
        padded = np.pad(color_mask, ((1, 1), (1, 1)), mode='constant')
        contours = (
            (padded[:-2, 1:-1] != padded[2:, 1:-1]) | 
            (padded[1:-1, :-2] != padded[1:-1, 2:])
        ) & color_mask
        
        all_contours_mask |= contours
    
    # Рисуем ВСЕ контуры серым цветом
    canvas[all_contours_mask] = [180, 180, 180]
    
    # Теперь расставляем номера
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 11)
    except:
        font = ImageFont.load_default()
    
    placed_positions = []
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2)
        
        # Находим связные регионы
        regions = find_connected_regions(color_mask, min_region_size)
        
        for region_x, region_y, cx, cy in regions:
            # Проверяем, не слишком ли близко к другим номерам
            too_close = False
            for px, py in placed_positions:
                if math.sqrt((cx - px) ** 2 + (cy - py) ** 2) < 25:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Проверяем, что центр внутри области
            if not color_mask[cy, cx]:
                # Создаём мини-маску и находим центр
                mini_mask = np.zeros((h, w), dtype=bool)
                for rx, ry in zip(region_x, region_y):
                    mini_mask[ry, rx] = True
                cx, cy = get_center_in_region(mini_mask)
            
            # Рисуем номер
            num_str = str(color_idx + 1)
            bbox = draw.textbbox((0, 0), num_str, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Белый фон под номером
            padding = 2
            draw.rectangle(
                [cx - text_w // 2 - padding, cy - text_h // 2 - padding,
                 cx + text_w // 2 + padding, cy + text_h // 2 + padding],
                fill='white',
                outline=None
            )
            
            # Рисуем номер
            draw.text((cx - text_w // 2, cy - text_h // 2), num_str, fill=(0, 0, 0), font=font)
            placed_positions.append((cx, cy))
    
    return pil_canvas


def create_palette_image(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """Создание изображения палитры"""
    n_colors = len(palette)
    palette_width = 300
    square_size = 30
    palette_height = 70 + n_colors * 35
    
    palette_img = Image.new('RGB', (palette_width, palette_height), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 12)
        title_font = ImageFont.truetype("Arial.ttf", 15)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    palette_draw.text((10, 12), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=title_font)
    palette_draw.text((10, 33), f"Всего цветов: {n_colors}", fill='gray', font=font)
    
    for idx, color in enumerate(palette, start=1):
        y_pos = 55 + (idx - 1) * 35
        
        palette_draw.rectangle(
            [(12, y_pos), (40, y_pos + 26)],
            fill=color,
            outline=(180, 180, 180),
            width=1
        )
        
        palette_draw.text((50, y_pos + 4), f"{idx}.", fill='black', font=font)
        
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        palette_draw.text((90, y_pos + 4), hex_color, fill='gray', font=font)
        
        r = 7
        palette_draw.ellipse(
            [265 - r, y_pos + 13 - r, 265 + r, y_pos + 13 + r],
            fill=color,
            outline=(150, 150, 150),
            width=1
        )
    
    return palette_img


def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS,
                               min_region_size: int = MIN_REGION_SIZE) -> Tuple[io.BytesIO, io.BytesIO]:
    """Основная функция обработки"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image)
    
    quantized, labels, palette = cluster_colors(img_array, n_colors)
    coloring_img = create_coloring_page(img_array, quantized, labels, palette, min_region_size)
    palette_img = create_palette_image(palette)
    
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG', dpi=(300, 300))
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# Функции бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам"</b>\n\n'
        'Отправьте фото — получите раскраску!\n\n'
        '<b>Команды:</b>\n'
        '• <code>/colors 24</code> — цветов (3-48)\n'
        '• <code>/detail 150</code> — мин. область (50-500)\n'
        '• <code>/help</code> — справка',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/colors 24</code>', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 48:
        await update.message.reply_text('❌ 3-48 цветов', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ {n_colors} цветов')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/detail 150</code>', parse_mode='HTML')
        return
    min_size = int(context.args[0])
    if not 50 <= min_size <= 500:
        await update.message.reply_text('❌ 50-500', parse_mode='HTML')
        return
    context.user_data['min_size'] = min_size
    await update.message.reply_text(f'✅ Мин. область: {min_size}px')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
    status_msg = await message.reply_text('🎨 Создаю раскраску...')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        min_size = context.user_data.get('min_size', MIN_REGION_SIZE)
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_size
        )
        
        await message.reply_photo(coloring_buffer, caption=f'🖼️ Раскраска!\n🎨 Цветов: {n_colors}')
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка</b>\n\n'
        '<b>Команды:</b>\n'
        '• /start — начать\n'
        '• /colors N — цветов (3-48)\n'
        '• /detail N — мин. область (50-500)\n'
        '• /help — справка',
        parse_mode='HTML'
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('detail', set_detail))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 Бот запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
