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

# Настройки по умолчанию
DEFAULT_N_COLORS = 24
DEFAULT_MIN_REGION_SIZE = 150
DEFAULT_MAX_IMAGE_SIZE = 1500
DEFAULT_LINE_THICKNESS = 1  # Толщина линии: 1, 2 или 3
DEFAULT_LINE_COLOR = 'gray'  # Цвет линий: gray, dark, light
DEFAULT_FONT_SIZE = 11      # Размер шрифта номеров: 9, 10, 11, 12, 13, 14
DEFAULT_PREPROCESS_STRENGTH = 'medium'  # Сила предобработки: light, medium, strong

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int, strength: str = 'medium') -> np.ndarray:
    """Предобработка изображения с настраиваемой силой"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Настройка силы фильтрации
    if strength == 'light':
        d, sigma = 5, 30
    elif strength == 'medium':
        d, sigma = 7, 50
    else:  # strong
        d, sigma = 9, 75
    
    img_array = cv2.bilateralFilter(img_array, d, sigma, sigma)
    
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

def merge_small_regions(labels: np.ndarray, palette_size: int, 
                        min_dist: int = 3, min_area: int = 100) -> np.ndarray:
    """Объединяет близкие области одного цвета с помощью морфологии"""
    h, w = labels.shape
    new_labels = labels.copy()
    
    # Применяем морфологическую операцию "Закрытие" (Closing) для каждого цвета
    for color_idx in range(palette_size):
        mask = (labels == color_idx).astype(np.uint8) * 255
        
        # 1. Закрытие: убирает мелкие дырочки и объединяет соседние островки
        kernel = np.ones((min_dist, min_dist), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. Открытие: убирает мелкий шум
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Фильтрация по площади с помощью ConnectedComponents
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels_img == i] = 0
        
        # Эта маска заменяет пятна. Остальные пиксели будут позже заполнены.
        new_labels[mask > 0] = color_idx
        
    return new_labels
                            
def find_connected_regions(mask: np.ndarray, min_size: int = 100) -> List[Tuple[List[int], List[int], int, int]]:
    """Поиск связных областей через BFS"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
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


def create_coloring_page(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                        min_region_size: int, line_thickness: int, line_color: str,
                        font_size: int) -> Image.Image:
    """Создание раскраски с настраиваемыми параметрами"""
    h, w = quantized.shape[:2]
    
    # Определяем цвет линий
    color_map = {
        'gray': [180, 180, 180],
        'dark': [100, 100, 100],
        'light': [210, 210, 210]
    }
    line_rgb = color_map.get(line_color, [180, 180, 180])
    
    # Создаём холст
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Находим все контуры
    all_contours_mask = np.zeros((h, w), dtype=bool)
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2)
        
        # Контуры через разницу
        padded = np.pad(color_mask, ((1, 1), (1, 1)), mode='constant')
        contours = (
            (padded[:-2, 1:-1] != padded[2:, 1:-1]) | 
            (padded[1:-1, :-2] != padded[1:-1, 2:])
        ) & color_mask
        
        all_contours_mask |= contours
    
    # Применяем толщину линии
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        all_contours_mask = cv2.dilate(all_contours_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # Рисуем контуры
    canvas[all_contours_mask] = line_rgb
    
    # Расставляем номера
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    placed_positions = []
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2)
        regions = find_connected_regions(color_mask, min_region_size)
        
        for region_x, region_y, cx, cy in regions:
            # Проверяем расстояние до других номеров
            too_close = False
            for px, py in placed_positions:
                if math.sqrt((cx - px) ** 2 + (cy - py) ** 2) < font_size * 2.5:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Проверяем, что центр внутри области
            if not color_mask[cy, cx]:
                mini_mask = np.zeros((h, w), dtype=bool)
                for rx, ry in zip(region_x, region_y):
                    mini_mask[ry, rx] = True
                ys, xs = np.where(mini_mask)
                if len(ys) > 0:
                    cx, cy = xs[len(xs)//2], ys[len(ys)//2]
                else:
                    continue
            
            # Рисуем номер
            num_str = str(color_idx + 1)
            bbox = draw.textbbox((0, 0), num_str, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            padding = 2
            draw.rectangle(
                [cx - text_w // 2 - padding, cy - text_h // 2 - padding,
                 cx + text_w // 2 + padding, cy + text_h // 2 + padding],
                fill='white',
                outline=None
            )
            
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


def process_image_for_coloring(photo_bytes: bytes, n_colors: int, min_region_size: int,
                               line_thickness: int, line_color: str, font_size: int,
                               max_image_size: int, preprocess_strength: str) -> Tuple[io.BytesIO, io.BytesIO]:
    """Основная функция обработки со всеми параметрами"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image, max_image_size, preprocess_strength)
    
    quantized, labels, palette = cluster_colors(img_array, n_colors)
    labels = merge_small_regions(labels, len(palette), min_dist=4, min_area=int(min_region_size * 0.7))
    coloring_img = create_coloring_page(quantized, palette, min_region_size, line_thickness, line_color, font_size)
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
        '<b>Основные команды:</b>\n'
        '• <code>/colors 24</code> — количество цветов (3-48)\n'
        '• <code>/detail 150</code> — мин. размер области (50-500)\n\n'
        '<b>Тонкая настройка:</b>\n'
        '• <code>/line 1</code> — толщина линий (1-3)\n'
        '• <code>/linecolor gray</code> — цвет линий (gray/dark/light)\n'
        '• <code>/font 11</code> — размер шрифта (9-14)\n'
        '• <code>/size 1500</code> — макс. размер изображения (800-3000)\n'
        '• <code>/smooth medium</code> — сглаживание (light/medium/strong)\n'
        '• <code>/settings</code> — показать текущие настройки\n'
        '• <code>/help</code> — подробная справка',
        parse_mode='HTML'
    )


async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать текущие настройки"""
    settings_text = (
        '⚙️ <b>Текущие настройки:</b>\n\n'
        f'🎨 Цветов: {context.user_data.get("n_colors", DEFAULT_N_COLORS)}\n'
        f'📏 Мин. область: {context.user_data.get("min_region_size", DEFAULT_MIN_REGION_SIZE)}px\n'
        f'📝 Толщина линий: {context.user_data.get("line_thickness", DEFAULT_LINE_THICKNESS)}\n'
        f'🎨 Цвет линий: {context.user_data.get("line_color", DEFAULT_LINE_COLOR)}\n'
        f'🔤 Размер шрифта: {context.user_data.get("font_size", DEFAULT_FONT_SIZE)}\n'
        f'🖼️ Макс. размер: {context.user_data.get("max_image_size", DEFAULT_MAX_IMAGE_SIZE)}px\n'
        f'🌊 Сглаживание: {context.user_data.get("preprocess_strength", DEFAULT_PREPROCESS_STRENGTH)}'
    )
    await update.message.reply_text(settings_text, parse_mode='HTML')


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/colors 24</code> (3-48)', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 48:
        await update.message.reply_text('❌ Допустимый диапазон: 3-48 цветов', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ Установлено {n_colors} цветов')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/detail 150</code> (50-500)', parse_mode='HTML')
        return
    min_size = int(context.args[0])
    if not 50 <= min_size <= 500:
        await update.message.reply_text('❌ Допустимый диапазон: 50-500 px', parse_mode='HTML')
        return
    context.user_data['min_region_size'] = min_size
    await update.message.reply_text(f'✅ Мин. область: {min_size}px')


async def set_line_thickness(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Настройка толщины линий"""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/line 1</code> (1-3)\n'
                                      '1 = тонкие, 2 = средние, 3 = толстые', parse_mode='HTML')
        return
    thickness = int(context.args[0])
    if not 1 <= thickness <= 3:
        await update.message.reply_text('❌ Допустимый диапазон: 1-3', parse_mode='HTML')
        return
    context.user_data['line_thickness'] = thickness
    names = {1: 'тонкие', 2: 'средние', 3: 'толстые'}
    await update.message.reply_text(f'✅ Толщина линий: {names[thickness]}')


async def set_line_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Настройка цвета линий"""
    if not context.args or context.args[0].lower() not in ['gray', 'dark', 'light']:
        await update.message.reply_text('❌ Используйте: <code>/linecolor gray</code>\n'
                                      'Варианты: gray, dark, light', parse_mode='HTML')
        return
    color = context.args[0].lower()
    context.user_data['line_color'] = color
    names = {'gray': 'серые', 'dark': 'тёмные', 'light': 'светлые'}
    await update.message.reply_text(f'✅ Цвет линий: {names[color]}')


async def set_font_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Настройка размера шрифта номеров"""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/font 11</code> (9-14)', parse_mode='HTML')
        return
    size = int(context.args[0])
    if not 9 <= size <= 14:
        await update.message.reply_text('❌ Допустимый диапазон: 9-14', parse_mode='HTML')
        return
    context.user_data['font_size'] = size
    await update.message.reply_text(f'✅ Размер шрифта: {size}')


async def set_max_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Настройка максимального размера изображения"""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/size 1500</code> (800-3000)\n'
                                      'Больше = детальнее, но медленнее', parse_mode='HTML')
        return
    size = int(context.args[0])
    if not 800 <= size <= 3000:
        await update.message.reply_text('❌ Допустимый диапазон: 800-3000', parse_mode='HTML')
        return
    context.user_data['max_image_size'] = size
    await update.message.reply_text(f'✅ Макс. размер: {size}px')


async def set_smooth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Настройка силы сглаживания"""
    if not context.args or context.args[0].lower() not in ['light', 'medium', 'strong']:
        await update.message.reply_text('❌ Используйте: <code>/smooth medium</code>\n'
                                      'Варианты: light, medium, strong\n'
                                      'light = больше деталей, strong = больше сглаживания', parse_mode='HTML')
        return
    strength = context.args[0].lower()
    context.user_data['preprocess_strength'] = strength
    names = {'light': 'слабое (больше деталей)', 'medium': 'среднее', 'strong': 'сильное (больше сглаживания)'}
    await update.message.reply_text(f'✅ Сглаживание: {names[strength]}')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
    # Получаем все настройки
    n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
    min_region_size = context.user_data.get('min_region_size', DEFAULT_MIN_REGION_SIZE)
    line_thickness = context.user_data.get('line_thickness', DEFAULT_LINE_THICKNESS)
    line_color = context.user_data.get('line_color', DEFAULT_LINE_COLOR)
    font_size = context.user_data.get('font_size', DEFAULT_FONT_SIZE)
    max_image_size = context.user_data.get('max_image_size', DEFAULT_MAX_IMAGE_SIZE)
    preprocess_strength = context.user_data.get('preprocess_strength', DEFAULT_PREPROCESS_STRENGTH)
    
    settings_text = (
        f'🎨 Цветов: {n_colors} | 📏 Мин: {min_region_size}px\n'
        f'📝 Линии: {line_thickness} | 🎨 Цвет: {line_color}\n'
        f'🔤 Шрифт: {font_size} | 🌊 Сглаж: {preprocess_strength}'
    )
    
    status_msg = await message.reply_text(f'🎨 Создаю раскраску...\n\n{settings_text}')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_region_size,
            line_thickness, line_color, font_size, max_image_size, preprocess_strength
        )
        
        await message.reply_photo(coloring_buffer, caption=f'🖼️ Раскраска!\n{settings_text}')
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте другие настройки.')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Полная справка</b>\n\n'
        '<b>Основные команды:</b>\n'
        '• <code>/colors N</code> — цветов (3-48, по умолч. 24)\n'
        '  Больше цветов = больше деталей\n\n'
        '• <code>/detail N</code> — мин. область (50-500, по умолч. 150)\n'
        '  Меньше = больше мелких зон\n\n'
        '<b>Тонкая настройка:</b>\n'
        '• <code>/line N</code> — толщина линий (1-3, по умолч. 1)\n'
        '  1 = тонкие, 2 = средние, 3 = толстые\n\n'
        '• <code>/linecolor X</code> — цвет линий\n'
        '  gray = серые, dark = тёмные, light = светлые\n\n'
        '• <code>/font N</code> — размер шрифта (9-14, по умолч. 11)\n\n'
        '• <code>/size N</code> — макс. размер (800-3000, по умолч. 1500)\n'
        '  Больше = детальнее, но медленнее\n\n'
        '• <code>/smooth X</code> — сглаживание\n'
        '  light = слабое, medium = среднее, strong = сильное\n\n'
        '• <code>/settings</code> — показать текущие настройки\n\n'
        '<b>Рекомендации:</b>\n'
        '• Для портретов: /colors 30 /detail 100 /smooth light\n'
        '• Для пейзажей: /colors 20 /detail 200 /smooth medium\n'
        '• Тонкие линии: /line 1 /linecolor light',
        parse_mode='HTML'
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('settings', show_settings))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('detail', set_detail))
    application.add_handler(CommandHandler('line', set_line_thickness))
    application.add_handler(CommandHandler('linecolor', set_line_color))
    application.add_handler(CommandHandler('font', set_font_size))
    application.add_handler(CommandHandler('size', set_max_size))
    application.add_handler(CommandHandler('smooth', set_smooth))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 Бот запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
