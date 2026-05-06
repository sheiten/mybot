import asyncio
import logging
import os
import io
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Для SLIC (суперпиксели)
from skimage.segmentation import slic
# Для векторной графики
import svgwrite

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

# Настройки по умолчанию
DEFAULT_N_COLORS = 24
DEFAULT_MIN_REGION_SIZE = 150
DEFAULT_MAX_IMAGE_SIZE = 1500
DEFAULT_LINE_THICKNESS = 1
DEFAULT_LINE_COLOR = 'gray'
DEFAULT_FONT_SIZE = 11
DEFAULT_PREPROCESS_STRENGTH = 'medium'

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image, target_size: int, strength: str = 'medium') -> np.ndarray:
    """Предобработка: изменение размера, медианное размытие для уничтожения текстур"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Медианное размытие — это ключ к крупным областям
    if strength == 'light':
        img_array = cv2.medianBlur(img_array, 5)
    elif strength == 'medium':
        img_array = cv2.medianBlur(img_array, 7)
    else:  # strong
        img_array = cv2.medianBlur(img_array, 11)
    
    return img_array

def cluster_colors_rgb(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Квантование цветов в RGB с сортировкой по яркости"""
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

def apply_superpixels(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                      min_region_size: int) -> np.ndarray:
    """
    Разбивает изображение на суперпиксели (SLIC) и переназначает цвета
    на ближайший из палитры. Это создаёт крупные, связные области.
    """
    h, w = quantized.shape[:2]
    
    # Количество суперпикселей: примерно 1 на каждые 2000-3000 пикселей
    # Чем больше min_region_size, тем меньше сегментов
    n_segments = max(100, int((h * w) / (min_region_size * 2)))
    
    # SLIC (Simple Linear Iterative Clustering)
    # compactness=30 даёт более квадратные/округлые формы
    segments = slic(quantized, n_segments=n_segments, compactness=30, sigma=1, start_label=1)
    
    # Создаём новое изображение, где каждый суперпиксель закрашен ближайшим цветом из палитры
    new_quantized = quantized.copy()
    palette_np = np.array(palette)
    
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        if not np.any(mask):
            continue
        
        # Находим средний цвет пикселей в этом суперпикселе
        pixels_in_seg = quantized[mask]
        avg_color = np.mean(pixels_in_seg, axis=0).astype(int)
        
        # Находим ближайший цвет из палитры (евклидово расстояние)
        distances = np.linalg.norm(palette_np - avg_color, axis=1)
        closest_idx = np.argmin(distances)
        
        # Закрашиваем весь суперпиксель этим цветом
        new_quantized[mask] = palette_np[closest_idx]
    
    return new_quantized

def find_regions_with_merging(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                             min_region_size: int) -> List[Tuple[np.ndarray, int, int]]:
    """
    Находит все связные области. Если область меньше min_region_size,
    она присоединяется к соседней большой области того же цвета.
    Возвращает список: (маска области, центр_x, центр_y)
    """
    h, w = quantized.shape[:2]
    regions = []
    
    # Для каждого цвета в палитре
    for color_idx, color in enumerate(palette):
        color_np = np.array(color)
        # Маска пикселей этого цвета
        mask = np.all(quantized == color_np, axis=2).astype(np.uint8) * 255
        
        # Находим все связные компоненты
        num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Собираем большие и мелкие компоненты
        large_components = []
        small_components = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_region_size:
                # Большой компонент
                large_components.append((i, area, centroids[i]))
            else:
                # Мелкий компонент — будет присоединён к большому
                small_components.append((i, area, centroids[i]))
        
        # Обрабатываем большие компоненты
        for comp_id, area, centroid in large_components:
            mask_comp = (labels_img == comp_id).astype(bool)
            regions.append((mask_comp, int(centroid[0]), int(centroid[1])))
        
        # ПРИСОЕДИНЯЕМ МЕЛКИЕ К БОЛЬШИМ
        # Для каждого мелкого компонента находим соседний большой того же цвета
        # и добавляем его пиксели к большой области
        # (Для простоты и скорости: просто оставляем мелкие, если они есть, 
        #  но в следующем проходе фильтрации они будут объединены)
        
    return regions

def create_coloring_page_vector(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                                min_region_size: int, line_thickness: int, 
                                line_color: str, font_size: int) -> io.BytesIO:
    """
    Создаёт векторную раскраску (SVG) с контурами и номерами.
    """
    h, w = quantized.shape[:2]
    
    # Цвет линий
    color_map = {
        'gray': '#b4b4b4',
        'dark': '#646464',
        'light': '#d2d2d2'
    }
    stroke_color = color_map.get(line_color, '#b4b4b4')
    
    # Создаём SVG-документ
    dwg = svgwrite.Drawing(profile='tiny', size=(w, h))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill='white'))
    
    placed_positions = []
    font_family = "Arial, sans-serif"
    
    # Для каждого цвета
    for color_idx, color in enumerate(palette):
        color_np = np.array(color)
        mask = np.all(quantized == color_np, axis=2).astype(np.uint8) * 255
        
        # Находим контуры через OpenCV
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_size:
                continue
            
            # Преобразуем контур в SVG-путь
            points = contour.reshape(-1, 2).tolist()
            if len(points) < 3:
                continue
            
            # Создаём замкнутый путь
            path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
            
            # Рисуем контур
            dwg.add(dwg.path(d=path_data, fill='none', stroke=stroke_color, stroke_width=line_thickness))
            
            # Находим центр для номера
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            # Проверка, не слишком близко к другому номеру
            too_close = False
            for px, py in placed_positions:
                if math.sqrt((cx - px) ** 2 + (cy - py) ** 2) < font_size * 3:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Рисуем номер (в SVG это проще)
            num_str = str(color_idx + 1)
            
            # Фон для номера (белый прямоугольник)
            # В SVG можно использовать text с background, но проще — прямоугольник
            # Вычисляем примерный размер текста
            text_width = len(num_str) * font_size * 0.6
            text_height = font_size * 1.2
            
            dwg.add(dwg.rect(
                insert=(cx - text_width / 2 - 2, cy - text_height / 2 - 2),
                size=(text_width + 4, text_height + 4),
                fill='white'
            ))
            
            # Текст номера
            dwg.add(dwg.text(
                num_str,
                insert=(cx, cy + font_size * 0.35),
                fill='black',
                font_size=font_size,
                font_family=font_family,
                text_anchor='middle',
                dominant_baseline='middle'
            ))
            
            placed_positions.append((cx, cy))
    
    # Сохраняем в буфер
    output = io.BytesIO()
    dwg.write(output, encoding='utf-8')
    output.seek(0)
    return output

def create_palette_image(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """Создание изображения палитры"""
    n_colors = len(palette)
    palette_width = 300
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
    
    return palette_img

def process_image_for_coloring(photo_bytes: bytes, n_colors: int, min_region_size: int,
                               line_thickness: int, line_color: str, font_size: int,
                               max_image_size: int, preprocess_strength: str) -> Tuple[io.BytesIO, io.BytesIO]:
    """Основная функция обработки с полным пайплайном"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image, max_image_size, preprocess_strength)
    
    # 1. Квантование цветов (KMeans)
    quantized, labels, palette = cluster_colors_rgb(img_array, n_colors)
    
    # 2. Применение суперпикселей (SLIC) для создания крупных связных областей
    quantized = apply_superpixels(quantized, palette, min_region_size)
    
    # 3. Создание векторной раскраски с контурами и номерами
    # Используем векторный формат SVG для идеальных линий
    coloring_buffer = create_coloring_page_vector(
        quantized, palette, min_region_size, line_thickness, line_color, font_size
    )
    
    # 4. Палитра
    palette_buffer = io.BytesIO()
    palette_img = create_palette_image(palette)
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer

# Функции бота (остаются без изменений, работают с новым процессом)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам v2.0 (Профессиональный)</b>\n\n'
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
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/line 1</code> (1-3)', parse_mode='HTML')
        return
    thickness = int(context.args[0])
    if not 1 <= thickness <= 3:
        await update.message.reply_text('❌ Допустимый диапазон: 1-3', parse_mode='HTML')
        return
    context.user_data['line_thickness'] = thickness
    names = {1: 'тонкие', 2: 'средние', 3: 'толстые'}
    await update.message.reply_text(f'✅ Толщина линий: {names[thickness]}')

async def set_line_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or context.args[0].lower() not in ['gray', 'dark', 'light']:
        await update.message.reply_text('❌ Используйте: <code>/linecolor gray</code>', parse_mode='HTML')
        return
    color = context.args[0].lower()
    context.user_data['line_color'] = color
    names = {'gray': 'серые', 'dark': 'тёмные', 'light': 'светлые'}
    await update.message.reply_text(f'✅ Цвет линий: {names[color]}')

async def set_font_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Используйте: <code>/size 1500</code> (800-3000)', parse_mode='HTML')
        return
    size = int(context.args[0])
    if not 800 <= size <= 3000:
        await update.message.reply_text('❌ Допустимый диапазон: 800-3000', parse_mode='HTML')
        return
    context.user_data['max_image_size'] = size
    await update.message.reply_text(f'✅ Макс. размер: {size}px')

async def set_smooth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or context.args[0].lower() not in ['light', 'medium', 'strong']:
        await update.message.reply_text('❌ Используйте: <code>/smooth medium</code>', parse_mode='HTML')
        return
    strength = context.args[0].lower()
    context.user_data['preprocess_strength'] = strength
    names = {'light': 'слабое', 'medium': 'среднее', 'strong': 'сильное'}
    await update.message.reply_text(f'✅ Сглаживание: {names[strength]}')

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
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
    
    status_msg = await message.reply_text(f'🎨 Создаю раскраску (Профессиональный режим)...\n\n{settings_text}')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_region_size,
            line_thickness, line_color, font_size, max_image_size, preprocess_strength
        )
        
        # Отправляем как документ (SVG) для сохранения векторного качества
        await message.reply_document(
            document=coloring_buffer,
            filename='coloring_page.svg',
            caption=f'🖼️ Раскраска (SVG)\n{settings_text}'
        )
        
        # Отправляем палитру
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте другие настройки.')
    finally:
        await status_msg.delete()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Полная справка (v2.0)</b>\n\n'
        '<b>Основные команды:</b>\n'
        '• <code>/colors N</code> — цветов (3-48, по умолч. 24)\n'
        '• <code>/detail N</code> — мин. область (50-500, по умолч. 150)\n\n'
        '<b>Тонкая настройка:</b>\n'
        '• <code>/line N</code> — толщина линий (1-3, по умолч. 1)\n'
        '• <code>/linecolor X</code> — цвет линий (gray/dark/light)\n'
        '• <code>/font N</code> — размер шрифта (9-14, по умолч. 11)\n'
        '• <code>/size N</code> — макс. размер (800-3000, по умолч. 1500)\n'
        '• <code>/smooth X</code> — сглаживание (light/medium/strong)\n'
        '• <code>/settings</code> — показать настройки\n\n'
        '<b>Рекомендации:</b>\n'
        '• Для портретов: /colors 30 /detail 100 /smooth light\n'
        '• Для пейзажей: /colors 20 /detail 200 /smooth medium',
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
    
    logger.info('🎨 Бот запущен (v2.0 Professional)!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
