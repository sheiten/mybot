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
MIN_REGION_SIZE = 200  # Увеличил, как в скрипте нейросети
MAX_IMAGE_SIZE = 1500

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Загрузка и предобработка изображения"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Сильное упрощение: убираем мелкие детали, сохраняя границы
    shifted = cv2.pyrMeanShiftFiltering(img_bgr, 20, 45)
    
    return shifted


def cluster_colors(img_bgr: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Квантование цветов через K-Means"""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")
    
    # Сортировка центров по яркости для красивой палитры
    brightness = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in centers])
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers[sorted_indices]
    
    # Переназначаем метки согласно сортировке
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels_mapped = np.array([label_map[l] for l in labels])
    
    # Квантованное изображение с отсортированными цветами
    quantized = centers_sorted[labels_mapped].reshape((h, w, 3))
    
    return quantized, centers_sorted, labels_mapped.reshape((h, w))


def create_coloring_page(quantized: np.ndarray, centers: np.ndarray, labels_map: np.ndarray, min_region_size: int) -> Tuple[Image.Image, Image.Image]:
    """Создание раскраски и палитры"""
    h, w = quantized.shape[:2]
    
    # Создание контурного холста через Canny
    gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 50)
    
    # Белый фон с серыми линиями
    canvas = np.ones((h, w, 3), dtype="uint8") * 255
    canvas[edges > 0] = [180, 180, 180]  # Светло-серые границы
    
    # Конвертируем в PIL для работы с текстом
    coloring = Image.fromarray(canvas)
    draw = ImageDraw.Draw(coloring)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Расстановка номеров
    placed_positions = []
    
    for i, color in enumerate(centers):
        mask = cv2.inRange(quantized, color, color)
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
            
            # Проверяем расстояние до других номеров
            too_close = False
            for px, py in placed_positions:
                dist = math.sqrt((cX - px)**2 + (cY - py)**2)
                if dist < 25:
                    too_close = True
                    break
            
            if not too_close:
                num_str = str(i + 1)
                bbox = draw.textbbox((0, 0), num_str, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # Белый фон под номером
                draw.rectangle(
                    [cX - text_w//2 - 2, cY - text_h//2 - 2,
                     cX + text_w//2 + 2, cY + text_h//2 + 2],
                    fill='white',
                    outline=None
                )
                
                # Рисуем номер
                draw.text((cX - text_w//2, cY - text_h//2), num_str, fill=(0, 0, 0), font=font)
                placed_positions.append((cX, cY))
    
    # Создание палитры
    palette = create_palette(centers)
    
    return coloring, palette


def create_palette(centers: np.ndarray) -> Image.Image:
    """Создание изображения палитры"""
    n_colors = len(centers)
    palette_width = 300
    square_size = 30
    palette_height = 80 + n_colors * 40
    
    palette_img = Image.new('RGB', (palette_width, palette_height), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    palette_draw.text((10, 15), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=title_font)
    palette_draw.text((10, 38), f"Всего цветов: {n_colors}", fill='gray', font=font)
    
    for idx, color in enumerate(centers, start=1):
        y_pos = 65 + (idx - 1) * 40
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
            [255-r, y_pos+square_size//2-r, 255+r, y_pos+square_size//2+r],
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
        '• <code>/colors 12</code> — количество цветов (3-30)\n'
        '• <code>/detail 200</code> — мин. размер области (50-500)\n'
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
        await update.message.reply_text('❌ <code>/detail 200</code>', parse_mode='HTML')
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
