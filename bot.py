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

DEFAULT_N_COLORS = 24  # Увеличил для большей детализации
MIN_REGION_SIZE = 150
MAX_IMAGE_SIZE = 1500

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Улучшенная предобработка с сохранением деталей"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Увеличиваем контраст через CLAHE для сохранения деталей
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Мягкое шумоподавление с сохранением краёв
    img_bgr = cv2.bilateralFilter(img_bgr, 7, 50, 50)
    
    # 3. Умеренное упрощение — уменьшены параметры
    # sp=10 (было 20) — меньше сглаживания
    # sr=30 (было 45) — меньше цветового сглаживания
    shifted = cv2.pyrMeanShiftFiltering(img_bgr, 10, 30)
    
    return shifted


def cluster_colors(img_bgr: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Квантование цветов с улучшенной кластеризацией"""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))
    
    # Используем мини-батч K-Means для ускорения на больших изображениях
    from sklearn.cluster import MiniBatchKMeans
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=1000,
        n_init=3
    )
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")
    
    # Сортировка по яркости
    brightness = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in centers])
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers[sorted_indices]
    
    # Переназначаем метки
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels_mapped = np.array([label_map[l] for l in labels])
    
    # Квантованное изображение
    quantized = centers_sorted[labels_mapped].reshape((h, w, 3))
    
    return quantized, centers_sorted, labels_mapped.reshape((h, w))


def create_coloring_page(quantized: np.ndarray, centers: np.ndarray, labels_map: np.ndarray, min_region_size: int) -> Tuple[Image.Image, Image.Image]:
    """Создание раскраски с улучшенной детализацией и нумерацией"""
    h, w = quantized.shape[:2]
    
    # Создание контуров через Canny с адаптивными порогами
    gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
    
    # Адаптивные пороги для Canny на основе статистики изображения
    median_intensity = np.median(gray)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    
    edges = cv2.Canny(gray, lower, upper)
    
    # Утолщаем границы для лучшей видимости
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Белый фон
    canvas = np.ones((h, w, 3), dtype="uint8") * 255
    canvas[edges > 0] = [160, 160, 160]  # Серые границы
    
    coloring = Image.fromarray(canvas)
    draw = ImageDraw.Draw(coloring)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 11)
        small_font = ImageFont.truetype("Arial.ttf", 9)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Расстановка номеров во ВСЕ регионы
    placed_positions = []
    
    for i, color in enumerate(centers):
        # Создаём маску для текущего цвета
        lower = np.clip(color.astype(int) - 5, 0, 255)
        upper = np.clip(color.astype(int) + 5, 0, 255)
        mask = cv2.inRange(quantized, lower, upper)
        
        # Морфологическая очистка маски
        kernel_clean = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_region_size:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Проверка, что центр внутри контура
            if cv2.pointPolygonTest(cnt, (cX, cY), False) < 0:
                continue
            
            # Проверяем расстояние до других номеров
            too_close = False
            for px, py in placed_positions:
                dist = math.sqrt((cX - px)**2 + (cY - py)**2)
                if dist < 20:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Выбираем размер шрифта в зависимости от площади
            current_font = small_font if area < 500 else font
            
            num_str = str(i + 1)
            bbox = draw.textbbox((0, 0), num_str, font=current_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Белый фон под номером
            padding = 1
            draw.rectangle(
                [cX - text_w//2 - padding, cY - text_h//2 - padding,
                 cX + text_w//2 + padding, cY + text_h//2 + padding],
                fill='white',
                outline=None
            )
            
            # Рисуем номер
            draw.text(
                (cX - text_w//2, cY - text_h//2),
                num_str,
                fill=(0, 0, 0),
                font=current_font
            )
            placed_positions.append((cX, cY))
    
    # Создание палитры
    palette = create_palette(centers)
    
    return coloring, palette


def create_palette(centers: np.ndarray) -> Image.Image:
    """Создание изображения палитры"""
    n_colors = len(centers)
    palette_width = 320
    square_size = 30
    palette_height = 80 + n_colors * 38
    
    palette_img = Image.new('RGB', (palette_width, palette_height), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
        title_font = ImageFont.truetype("Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    palette_draw.text((10, 15), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=title_font)
    palette_draw.text((10, 38), f"Всего цветов: {n_colors}", fill='gray', font=font)
    
    for idx, color in enumerate(centers, start=1):
        y_pos = 65 + (idx - 1) * 38
        color_tuple = tuple(map(int, color))
        
        # Квадрат с цветом
        palette_draw.rectangle(
            [(15, y_pos), (15 + square_size, y_pos + square_size)],
            fill=color_tuple,
            outline=(200, 200, 200),
            width=1
        )
        
        # Номер и HEX
        palette_draw.text((55, y_pos + 5), f"{idx}.", fill='black', font=font)
        hex_color = f'#{color_tuple[0]:02x}{color_tuple[1]:02x}{color_tuple[2]:02x}'
        palette_draw.text((100, y_pos + 5), hex_color, fill='gray', font=font)
        
        # Кружок с цветом
        r = 8
        palette_draw.ellipse(
            [275-r, y_pos+square_size//2-r, 275+r, y_pos+square_size//2+r],
            fill=color_tuple,
            outline=(180, 180, 180),
            width=1
        )
    
    return palette_img


def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS, min_region_size: int = MIN_REGION_SIZE) -> Tuple[io.BytesIO, io.BytesIO]:
    """Основная функция обработки"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_bgr = preprocess_image(image)
    
    quantized, centers, labels_map = cluster_colors(img_bgr, n_colors)
    coloring_img, palette_img = create_coloring_page(quantized, centers, labels_map, min_region_size)
    
    # Сохраняем результат
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG', dpi=(300, 300))
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# Функции бота (без изменений)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам"</b>\n\n'
        'Отправьте фото — получите раскраску!\n\n'
        '<b>Команды:</b>\n'
        '• <code>/colors 24</code> — количество цветов (3-48)\n'
        '• <code>/detail 150</code> — мин. размер области (50-500)\n'
        '• <code>/help</code> — справка\n\n'
        '💡 Больше цветов = больше деталей',
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
    
    status_msg = await message.reply_text('🎨 Обрабатываю... Это может занять до 30 секунд')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        min_size = context.user_data.get('min_size', MIN_REGION_SIZE)
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_size
        )
        
        await message.reply_photo(coloring_buffer, caption=f'🖼️ Раскраска!\n🎨 Цветов: {n_colors}\n📏 Мин. область: {min_size}px')
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте другое фото.')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка</b>\n\n'
        '<b>Команды:</b>\n'
        '• /start — начать\n'
        '• /colors N — цветов (3-48, по умолч. 24)\n'
        '• /detail N — мин. область в px (50-500, по умолч. 150)\n'
        '• /help — справка\n\n'
        '<b>Советы:</b>\n'
        '• Больше цветов = больше деталей\n'
        '• Меньше мин. область = больше мелких зон\n'
        '• Для портретов: /colors 30 /detail 100\n'
        '• Для пейзажей: /colors 20 /detail 200',
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
