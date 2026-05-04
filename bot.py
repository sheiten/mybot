import asyncio
import logging
import os
import io
import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# === КОНФИГУРАЦИЯ ===
TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

DEFAULT_N_COLORS = 12
DEFAULT_MIN_SIZE = 500
MAX_IMAGE_SIZE = 800
FONT_SIZE = 24  # Чуть увеличил для читаемости

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Пытается загрузить шрифт, доступный в системе (особенно актуально для Linux VPS).
    Список включает наиболее распространенные шрифты в Linux дистрибутивах.
    """
    # Список шрифтов для проверки (от приоритетных к менее приоритетным)
    font_names = [
        "DejaVuSans.ttf",           # Стандарт в Ubuntu/Debian
        "FreeSans.ttf",             # Часто встречается
        "LiberationSans-Regular.ttf", # RHEL/CentOS
        "arial.ttf"                 # Windows/если установлен вручную
    ]
    
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
            
    # Если ничего не найдено, грузим стандартный (пиксельный)
    logger.warning("Системные шрифты не найдены, используется стандартный.")
    return ImageFont.load_default()

# === ОБРАБОТКА ИЗОБРАЖЕНИЙ ===

def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        # Используем BOX или HAMMING для более мягкого уменьшения без лишней резкости
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # 1. Сильный медианный фильтр уберет мелкие "островки" пикселей
    img_array = cv2.medianBlur(img_array, 5)
    
    # 2. Билатеральный фильтр сделает цветовые пятна более однородными
    img_array = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    
    return img_array

def cluster_colors(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    h, w, c = img_array.shape
    
    # ПЕРЕХОДИМ В Lab для лучшей кластеризации
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    pixels = lab_img.reshape(-1, c).astype(np.float32)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Возвращаем центры в RGB
    centers_lab = kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    
    # Сортировка по яркости
    brightness = [0.299*r + 0.587*g + 0.114*b for r, g, b in centers_rgb]
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers_rgb[sorted_indices]
    
    # Перемаппинг
    label_map = np.zeros(n_colors, dtype=np.uint8)
    for new, old in enumerate(sorted_indices):
        label_map[old] = new
    labels_mapped = label_map[labels].reshape(h, w)

    # УДАЛЕНИЕ ШУМА (вместо medianBlur на индексах)
    # Морфологическое закрытие уберет мелкие дырки внутри сегментов
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    labels_mapped = cv2.morphologyEx(labels_mapped, cv2.MORPH_OPEN, kernel)
    
    # Создаем финальное изображение для визуализации
    clustered_img = centers_sorted[labels_mapped]
    
    return clustered_img, [tuple(c) for c in centers_sorted], labels_mapped


def find_largest_inscribed_circle(mask: np.ndarray):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    return maxLoc, maxVal

def create_coloring_page(width: int, height: int, labels: np.ndarray, palette: List[Tuple[int, int, int]], min_size: int = DEFAULT_MIN_SIZE) -> Tuple[Image.Image, Image.Image]:
    coloring = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(coloring)
    
    # Используем нашу функцию для получения шрифта
    font = get_font(FONT_SIZE)
    
    unique_labels = np.unique(labels)
    regions_info = []
    
    logger.info(f"Начало отрисовки контуров. Областей: {len(unique_labels)}")
    
    for label_idx in unique_labels:
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == label_idx] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Фильтр мелкого шума
            if area < min_size:
                continue
            
            # Упрощение контура (аппроксимация)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            
            # Рисуем линию
            draw.line(points + [points[0]], fill="black", width=2)
            
            # Ищем место для цифры
            temp_mask = np.zeros(labels.shape, dtype=np.uint8)
            cv2.drawContours(temp_mask, [approx], -1, 255, -1)
            center, radius = find_largest_inscribed_circle(temp_mask)
            
            # Сохраняем информацию о регионе, чтобы нарисовать цифры потом (сверху)
            if radius > FONT_SIZE * 1.2:
                regions_info.append({
                    'num': int(label_idx) + 1, # Номер цвета в палитре (с 1)
                    'center': center,
                    'radius': radius,
                    'area': area
                })
    
    # Сортируем по площади (сначала большие), чтобы цифры крупных областей рисовались в приоритете
    regions_info.sort(key=lambda x: x['area'], reverse=True)
    
    drawn_positions = []
    
    for region in regions_info:
        cx, cy = region['center']
        num_str = str(region['num'])
        r = region['radius']
        
        # Проверка, чтобы цифры не налезали друг на друга
        too_close = False
        for dx, dy in drawn_positions:
            dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
            if dist < FONT_SIZE * 1.5:
                too_close = True
                break
        
        if not too_close and r > FONT_SIZE * 1.2:
            # Центрируем текст
            try:
                bbox = draw.textbbox((0, 0), num_str, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text((cx - text_w // 2, cy - text_h // 2), num_str, fill="black", font=font)
                drawn_positions.append((cx, cy))
            except Exception:
                # Фоллбэк, если что-то пошло не так с шрифтом
                draw.text((cx, cy), num_str, fill="black")

    # Генерация палитры
    n_colors = len(palette)
    # Вычисляем высоту палитры динамически
    palette_h = 60 + n_colors * 40
    palette_img = Image.new('RGB', (250, palette_h), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    palette_draw.text((10, 15), "ПАЛИТРА:", fill='black', font=font)
    
    for idx, color in enumerate(palette, start=1):
        y_pos = 50 + (idx - 1) * 40
        # Цветной квадратик
        palette_draw.rectangle([(15, y_pos), (45, y_pos + 30)], fill=color, outline='black', width=2)
        # Номер
        palette_draw.text((55, y_pos + 10), f"{idx}", fill='black', font=font)
    
    return coloring, palette_img

def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS, min_size: int = DEFAULT_MIN_SIZE) -> Tuple[io.BytesIO, io.BytesIO]:
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image)
    h, w = img_array.shape[:2]
    
    _, palette, labels = cluster_colors(img_array, n_colors)
    coloring_img, palette_img = create_coloring_page(w, h, labels, palette, min_size)
    
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG', dpi=(300, 300))
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# === ОБРАБОТЧИКИ ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам" PRO</b>\n\n'
        'Профессиональное качество контуров.\n\n'
        '⚙️ <b>Команды:</b>\n'
        '• <code>/colors 12</code> — количество цветов (3-30)\n'
        '• <code>/detail high</code> — больше деталей\n'
        '• <code>/detail low</code> — чище, без мелочи\n'
        '• <code>/help</code> — справка',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ <code>/colors 12</code> (3-30)', parse_mode='HTML')
        return
    
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 30:
        await update.message.reply_text('❌ Разрешено от 3 до 30 цветов', parse_mode='HTML')
        return
    
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ Количество цветов установлено: {n_colors}')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text('❌ <code>/detail high</code> или <code>/detail low</code>', parse_mode='HTML')
        return # ИСПРАВЛЕНО: Return теперь внутри блока if
    
    level = context.args[0].lower()
    if level == 'high':
        # Меньше min_size = больше мелких деталей сохраняется
        context.user_data['min_size'] = 500
        await update.message.reply_text('✅ Высокая детализация (много мелких деталей)')
    elif level == 'low':
        # Больше min_size = мелкий шум убирается
        context.user_data['min_size'] = 5000
        await update.message.reply_text('✅ Низкая детализация (только крупные области)')
    else:
        await update.message.reply_text('❌ Используйте <code>high</code> или <code>low</code>', parse_mode='HTML')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    status_msg = await message.reply_text('⏳ Обрабатываю изображение... (может занять до 30 сек)')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        min_size = context.user_data.get('min_size', DEFAULT_MIN_SIZE)
        
        # Запуск тяжелой обработки в отдельном потоке, чтобы не "лагал" бот
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_size
        )
        
        await message.reply_photo(
            coloring_buffer, 
            caption=f'🖼️ <b>Раскраска готова!</b>\n🎨 Цветов: {n_colors}'
        )
        await message.reply_photo(palette_buffer, caption='🎨 Палитра цветов')
        
    except Exception as e:
        logger.error(f'Ошибка при обработке изображения: {e}', exc_info=True)
        await message.reply_text('❌ Произошла ошибка при обработке. Попробуйте отправить другое фото.')
    finally:
        try:
            await status_msg.delete()
        except:
            pass


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка</b>\n\n'
        '• /start — начать\n'
        '• /colors N — установить количество цветов (3-30)\n'
        '• /detail high|low — уровень детализации\n'
        '• /help — эта справка\n\n'
        '💡 <b>Советы:</b>\n'
        '- Для сложных фото с людьми используйте /colors 15-20 и /detail high.\n'
        '- Для простых пейзажей используйте /colors 8-12 и /detail low.',
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
