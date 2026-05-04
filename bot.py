import asyncio
import logging
import os
import io
from typing import List, Tuple
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

DEFAULT_N_COLORS = 18
MIN_REGION_SIZE = 100
MAX_IMAGE_SIZE = 1500
FONT_SIZE = 12

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Улучшенная предобработка: легкое сглаживание + повышение контраста
    img_array = cv2.bilateralFilter(img_array, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Повышение контраста для лучшего разделения цветов
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img_array


def cluster_colors(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # Сортировка по яркости (используем правильную формулу)
    brightness = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in centers])
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers[sorted_indices]
    
    # Ремаппинг меток
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels_mapped = np.array([label_map[l] for l in labels]).reshape(h, w).astype(np.uint8)
    
    # Медианный фильтр только один раз и меньшим ядром
    if n_colors > 10:
        labels_mapped = cv2.medianBlur(labels_mapped, 3)
    else:
        labels_mapped = cv2.medianBlur(labels_mapped, 5)
    
    # Удаляем очень маленькие области (острова)
    for label in np.unique(labels_mapped):
        mask = (labels_mapped == label).astype(np.uint8) * 255
        
        # Находим все компоненты связности для этого цвета
        num_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 30:  # Маленькие острова
                # Заменяем на цвет большинства соседей
                y, x = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2, \
                       stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
                
                # Получаем соседние метки
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and labels_mapped[ny, nx] != label:
                            neighbors.append(labels_mapped[ny, nx])
                
                if neighbors:
                    from collections import Counter
                    most_common = Counter(neighbors).most_common(1)[0][0]
                    labels_mapped[labels_mapped == label] = most_common
    
    return img_array, [tuple(c) for c in centers_sorted], labels_mapped


def find_largest_inscribed_circle(mask: np.ndarray):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    return maxLoc, maxVal


def create_coloring_page(width: int, height: int, labels: np.ndarray, palette: List[Tuple[int, int, int]], min_region_size: int) -> Tuple[Image.Image, Image.Image]:
    coloring = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(coloring)
    
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    
    # Создаем карту регионов по цветам
    regions_by_color = {}
    
    for label_idx in np.unique(labels):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == label_idx] = 255
        
        # Находим все контуры для этого цвета
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < min_region_size:
                continue
            
            # Адаптивное упрощение контуров
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.002 * perimeter  # Уменьшил упрощение для лучшей детализации
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            
            # Рисуем контур более толстой линией
            if len(points) > 2:
                draw.line(points + [points[0]], fill="black", width=2)
            
            # Сохраняем информацию о регионе
            if label_idx not in regions_by_color:
                regions_by_color[label_idx] = []
            
            regions_by_color[label_idx].append({
                'contour': approx,
                'area': area,
                'points': points
            })
    
    # Рисуем номера (по порядку цветов в палитре)
    drawn_positions = []
    
    for color_num, label_idx in enumerate(range(len(palette)), start=1):
        if label_idx not in regions_by_color:
            continue
        
        # Сортируем регионы по площади и берем самый большой для номера
        regions = sorted(regions_by_color[label_idx], key=lambda x: x['area'], reverse=True)
        
        for region in regions:
            if len(region['contour']) < 3:
                continue
            
            # Создаем маску для нахождения центра
            temp_mask = np.zeros(labels.shape, dtype=np.uint8)
            cv2.drawContours(temp_mask, [region['contour']], -1, 255, -1)
            
            center, radius = find_largest_inscribed_circle(temp_mask)
            
            # Проверяем, поместится ли цифра
            if radius > FONT_SIZE * 1.2:
                cx, cy = center
                
                # Проверяем расстояние до других цифр
                too_close = False
                for dx, dy, _ in drawn_positions:
                    dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
                    if dist < FONT_SIZE * 2.5:
                        too_close = True
                        break
                
                if not too_close:
                    num_str = str(color_num)
                    bbox = draw.textbbox((0, 0), num_str, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    
                    draw.text((cx - text_w // 2, cy - text_h // 2), num_str, fill="black", font=font)
                    drawn_positions.append((cx, cy, color_num))
                    break  # Ставим только одну цифру на цвет
    
    # Создаем палитру с улучшенным дизайном
    n_colors = len(palette)
    palette_height = 80 + n_colors * 45
    palette_img = Image.new('RGB', (300, palette_height), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    # Заголовок
    palette_draw.text((10, 15), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=font)
    palette_draw.text((10, 35), f"Всего цветов: {n_colors}", fill='gray', font=font)
    
    for idx, color in enumerate(palette, start=1):
        y_pos = 60 + (idx - 1) * 45
        
        # Квадрат с цветом
        palette_draw.rectangle([(15, y_pos), (50, y_pos + 35)], fill=color, outline='black', width=2)
        
        # Номер цвета
        palette_draw.text((65, y_pos + 10), f"{idx}", fill='black', font=font)
        
        # HEX-код цвета
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        palette_draw.text((100, y_pos + 10), hex_color, fill='gray', font=font)
        
        # Линия-разделитель
        if idx < n_colors:
            palette_draw.line([(15, y_pos + 40), (280, y_pos + 40)], fill='#e0e0e0', width=1)
    
    return coloring, palette_img


def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS, min_region_size: int = MIN_REGION_SIZE) -> Tuple[io.BytesIO, io.BytesIO]:
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image)
    h, w = img_array.shape[:2]
    
    _, palette, labels = cluster_colors(img_array, n_colors)
    coloring_img, palette_img = create_coloring_page(w, h, labels, palette, min_region_size)
    
    # Сохраняем с высоким качеством
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG', dpi=(300, 300), optimize=False)
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# Дальше идут функции бота - они НЕ ТРОГАЮТСЯ (только исправлен синтаксис в set_colors)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам"</b>\n\n'
        'Отправьте фото — получите раскраску!\n\n'
        '<b>Команды:</b>\n'
        '• <code>/colors 12</code> — количество цветов (3-30)\n'
        '• <code>/detail 100</code> — мин. размер области (50-500)\n'
        '• <code>/help</code> — справка',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/colors 12</code>', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 30:
        await update.message.reply_text('❌ 3-30 цветов', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ {n_colors} цветов')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/detail 100</code>', parse_mode='HTML')
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
    
    status_msg = await message.reply_text('🎨 Обрабатываю...')
    
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
        '• /colors N — цветов (3-30)\n'
        '• /detail N — мин. область (50-500)\n'
        '• /help — справка\n\n'
        '💡 Меньше /detail = больше деталей',
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
