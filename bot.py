import asyncio
import logging
import os
import io
from typing import List, Tuple, Dict
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# === КОНФИГУРАЦИЯ ===
TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

DEFAULT_N_COLORS = 12
MIN_REGION_SIZE = 3000  # Увеличил для чистоты
MAX_IMAGE_SIZE = 800    # Уменьшил для четкости
FONT_SIZE = 16

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Предобработка с сильным сглаживанием"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Ресайз
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Конвертация в numpy
    img_array = np.array(image)
    
    # Сильное сглаживание (Gaussian + Bilateral)
    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    img_array = cv2.bilateralFilter(img_array, d=9, sigmaColor=100, sigmaSpace=100)
        return img_array


def cluster_colors(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    """Кластеризация с пост-обработкой"""
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)
    
    # K-means
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # Сортировка по яркости
    brightness = [np.mean(color) for color in centers]
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers[sorted_indices]
    
    # Ремаппинг меток
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels_mapped = np.array([label_map[l] for l in labels]).reshape(h, w).astype(np.uint8)
    
    # КРИТИЧЕСКИ ВАЖНО: Морфологические операции для очистки
    # 1. Открытие (удаление мелких точек)
    kernel = np.ones((3,3),np.uint8)
    labels_mapped = cv2.morphologyEx(labels_mapped, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Закрытие (заполнение дыр)
    labels_mapped = cv2.morphologyEx(labels_mapped, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. Медианный фильтр (сглаживание границ)
    labels_mapped = cv2.medianBlur(labels_mapped, 5)
    
    return img_array, [tuple(c) for c in centers_sorted], labels_mapped


def find_largest_inscribed_circle(mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """Находит центр и радиус largest вписанной окружности в регионе"""
    # Расстояние до ближайшего фона
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, maxVal, maxLoc = cv2.minMaxLoc(dist)
    return maxLoc, maxVal


def create_coloring_page(width: int, height: int, labels: np.ndarray, palette: List[Tuple[int, int, int]]) -> Tuple[Image.Image, Image.Image]:
    """Создание раскраски с профессиональным качеством"""
    
    # Создаем белое изображение
    coloring = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(coloring)    
    # Загружаем шрифт
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    
    # Находим все уникальные метки
    unique_labels = np.unique(labels)
    regions_info = []
    
    for label_idx in unique_labels:
        # Создаем бинарную маску для текущего региона
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == label_idx] = 255
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # ФИЛЬТР: Только достаточно большие регионы
            if area < MIN_REGION_SIZE:
                continue
            
            # Сглаживание контура
            epsilon = 0.015 * cv2.arcLength(cnt, True)  # Более агрессивное упрощение
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Конвертация в список точек для PIL
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            
            # Рисуем контур (тонкая линия)
            draw.line(points + [points[0]], fill="black", width=2)
            
            # Находим оптимальное место для цифры
            # Создаем временную маску для этого контура
            temp_mask = np.zeros(labels.shape, dtype=np.uint8)
            cv2.drawContours(temp_mask, [approx], -1, 255, -1)
            
            # Находим центр largest вписанной окружности
            center, radius = find_largest_inscribed_circle(temp_mask)
            
            # Проверяем, что цифра поместится
            if radius > FONT_SIZE:
                regions_info.append({
                    'num': int(label_idx) + 1,
                    'center': center,
                    'radius': radius,                    'area': area
                })
    
    # Сортируем регионы по площади (большие primero)
    regions_info.sort(key=lambda x: x['area'], reverse=True)
    
    # Рисуем цифры с проверкой на наложения
    drawn_positions = []
    
    for region in regions_info:
        cx, cy = region['center']
        num_str = str(region['num'])
        r = region['radius']
        
        # Проверяем расстояние до уже нарисованных цифр
        too_close = False
        for dx, dy in drawn_positions:
            dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
            if dist < FONT_SIZE * 1.5:  # Минимальное расстояние между цифрами
                too_close = True
                break
        
        if not too_close and r > FONT_SIZE * 1.2:
            # Центрируем текст
            bbox = draw.textbbox((0, 0), num_str, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.text(
                (cx - text_w // 2, cy - text_h // 2), 
                num_str, 
                fill="black", 
                font=font
            )
            
            drawn_positions.append((cx, cy))
    
    # Создаем палитру
    n_colors = len(palette)
    palette_img = Image.new('RGB', (250, 60 + n_colors * 40), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    palette_draw.text((10, 15), "ПАЛИТРА ЦВЕТОВ:", fill='black', font=font)
    
    for idx, color in enumerate(palette, start=1):
        y_pos = 50 + (idx - 1) * 40
        # Квадрат цвета
        palette_draw.rectangle(
            [(15, y_pos), (45, y_pos + 30)], 
            fill=color,             outline='black', 
            width=2
        )
        # Номер
        palette_draw.text((55, y_pos + 10), f"{idx}", fill='black', font=font)
        # RGB (мелким шрифтом)
        r, g, b = color
        palette_draw.text(
            (80, y_pos + 12), 
            f"RGB({r},{g},{b})", 
            fill='gray', 
            font=ImageFont.load_default()
        )
    
    return coloring, palette_img


def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS) -> Tuple[io.BytesIO, io.BytesIO]:
    """Основная функция обработки"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image)
    h, w = img_array.shape[:2]
    
    _, palette, labels = cluster_colors(img_array, n_colors)
    coloring_img, palette_img = create_coloring_page(w, h, labels, palette)
    
    # Сохраняем в буферы
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG', dpi=(300, 300))
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# === ОБРАБОТЧИКИ TELEGRAM ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам" PRO</b>\n\n'
        'Профессиональное качество:\n'
        '• Гладкие контуры без зубцов\n'
        '• Цифры по центру областей\n'
        '• Без мелкого шума\n\n'
        '⚙️ <b>Команды:</b>\n'
        '• <code>/colors 12</code> — количество цветов (3-30)\n'
        '• <code>/detail high</code> — высокая детализация\n'        '• <code>/detail low</code> — меньше деталей, чище\n'
        '• <code>/help</code> — справка\n\n'
        '📤 Отправьте фото для начала!',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/colors 12</code> (3-30)', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 30:
        await update.message.reply_text('❌ 3-30 цветов', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ {n_colors} цветов установлено')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text('❌ <code>/detail high</code> или <code>/detail low</code>', parse_mode='HTML')
        return
    
    level = context.args[0].lower()
    if level == 'high':
        context.user_data['min_size'] = 1000
        await update.message.reply_text('✅ Высокая детализация (больше мелких деталей)')
    elif level == 'low':
        context.user_data['min_size'] = 5000
        await update.message.reply_text('✅ Низкая детализация (только крупные формы)')
    else:
        await update.message.reply_text('❌ Используйте <code>high</code> или <code>low</code>', parse_mode='HTML')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
    status_msg = await message.reply_text('🎨 Обрабатываю... (~15-30 сек)')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        min_size = context.user_data.get('min_size', MIN_REGION_SIZE)
        
        # Временно меняем глобальную переменную
        global MIN_REGION_SIZE        old_min_size = MIN_REGION_SIZE
        MIN_REGION_SIZE = min_size
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors
        )
        
        MIN_REGION_SIZE = old_min_size
        
        await message.reply_photo(
            coloring_buffer, 
            caption=f'🖼️ <b>Ваша раскраска!</b>\n'
                   f'🎨 Цветов: {n_colors}\n'
                   f'📐 Детализация: {"высокая" if min_size < 3000 else "низкая"}',
            parse_mode='HTML'
        )
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка. Попробуйте другое фото или /detail low')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка</b>\n\n'
        '<b>Команды:</b>\n'
        '• /start — начать\n'
        '• /colors N — цветов (3-30)\n'
        '• /detail high|low — детализация\n'
        '• /help — справка\n\n'
        '💡 <b>Советы:</b>\n'
        '• Для пейзажей: /colors 15-20\n'
        '• Для портретов: /colors 10-15\n'
        '• Если много шума: /detail low',
        parse_mode='HTML'
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('detail', set_detail))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
        logger.info('🎨 Бот PRO запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
