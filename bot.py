import asyncio
import logging
import os
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

# Настройки
MAX_IMAGE_SIZE = 1000
MIN_REGION_AREA = 250
NUM_COLORS = 24

DEFAULT_N_COLORS = NUM_COLORS
MIN_REGION_SIZE = MIN_REGION_AREA

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Упрощение изображения"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Билатеральный фильтр убирает мелкие детали, сохраняя края
    img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    # Легкое размытие для помощи K-means
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    
    return img_bgr


def apply_kmeans(img_bgr: np.ndarray, num_colors: int):
    """Квантование с очисткой от шума (секрет плавных зон)"""
    h, w = img_bgr.shape[:2]
    pixels = img_bgr.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")
    
    # МАГИЯ ФАБРИЧНОГО КАЧЕСТВА: медианный фильтр НА МЕТКАХ, а не на пикселях.
    # Это убирает эффект "соли и перца" (одиночные пиксели другого цвета)
    labels_2d = labels.reshape(h, w).astype(np.uint8)
    labels_clean = cv2.medianBlur(labels_2d, 5)
    
    quantized = centers[labels_clean].reshape(img_bgr.shape)
    return quantized, centers


def create_coloring_page(quantized: np.ndarray, centers: np.ndarray, min_region_size: int = 150):
    """Создание контурной карты фабричного качества"""
    h, w = quantized.shape[:2]
    
    # Используем PIL для антиалиасинга (гладких линий)
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Пытаемся загрузить красивый шрифт, если есть
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 10)
    except IOError:
        font = ImageFont.load_default()
        font_small = font

    for i, color in enumerate(centers):
        # 1. Создаем маску цвета
        mask = cv2.inRange(quantized, color, color)
        
        # 2. Морфологическое замыкание (убирает микро-дырки и соединяет разорванные линии)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 3. Ищем контуры С ИЕРАРХИЕЙ (чтобы были глаза, рот, внутренние детали)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            continue
            
        hierarchy = hierarchy[0]
        
        for j, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_region_size:
                continue
            
            # 4. ПЛАВНОСТЬ ЛИНИЙ: Адаптивная аппроксимация. 
            # 0.008 дает идеальные плавные кривые вместо зубчатых пикселей
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.008 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Конвертируем контур в список кортежей для PIL
            pil_contour = [tuple(pt[0]) for pt in approx]
            
            if len(pil_contour) < 3:
                continue
                
            # Рисуем контур (фабричная толщина - 2 пикселя, черный цвет)
            is_hole = hierarchy[j][3] != -1
            line_width = 2 if not is_hole else 1 # Внутренние детали чуть тоньше
            draw.line(pil_contour + [pil_contour[0]], fill=(0, 0, 0), width=line_width)
            
            # 5. УМНОЕ РАЗМЕЩЕНИЕ ЦИФР
            # Ставим цифру только на внешних контурах (не в дырках)
            if not is_hole:
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Проверяем, что центр точно внутри многоугольника
                    if cv2.pointPolygonTest(approx, (cX, cY), False) >= 0:
                        label = str(i + 1)
                        
                        # Выбираем размер шрифта в зависимости от площади
                        current_font = font if area > 1500 else font_small
                        
                        # Вычисляем размер текста для идеального центрирования
                        bbox = draw.textbbox((0, 0), label, font=current_font)
                        t_w = bbox[2] - bbox[0]
                        t_h = bbox[3] - bbox[1]
                        
                        draw.text(
                            (cX - t_w / 2, cY - t_h / 2), 
                            label, 
                            fill=(60, 60, 60), 
                            font=current_font
                        )
                            
    return np.array(canvas)


def create_palette_image(centers: np.ndarray, width: int) -> Image.Image:
    """Создание красивой палитры через PIL"""
    n_colors = len(centers)
    palette_height = 80
    palette = Image.new('RGB', (width, palette_height), (255, 255, 255))
    draw = ImageDraw.Draw(palette)
    
    swatch_w = width // n_colors
    margin = 4
    
    # Пытаемся использовать тот же шрифт
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for i, color in enumerate(centers):
        x_start = i * swatch_w
        
        # Рисуем квадратик цвета
        x1, y1 = x_start + margin, 10
        x2, y2 = x_start + swatch_w - margin, 50
        
        # ИСПРАВЛЕНО ТУТ: обернули в tuple()
        draw.rectangle([x1, y1, x2, y2], fill=tuple(color.tolist()), outline=(180, 180, 180), width=1)
        
        # Номер под квадратом
        label = str(i + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        t_w = bbox[2] - bbox[0]
        text_x = x_start + swatch_w // 2 - t_w // 2
        
        draw.text((text_x, 58), label, fill=(0, 0, 0), font=font)
    
    return palette


def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS,
                               min_region_size: int = MIN_REGION_SIZE):
    """Основная функция обработки"""
    input_img = Image.open(io.BytesIO(photo_bytes))
    
    simplified = preprocess_image(input_img)
    quantized, centers = apply_kmeans(simplified, n_colors)
    
    # Получаем numpy массив и конвертим в PIL
    canvas_np = create_coloring_page(quantized, centers, min_region_size)
    coloring_img = Image.fromarray(canvas_np)
    
    palette_img = create_palette_image(centers, coloring_img.width)
    
    # Сохраняем в память
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG')
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG')
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# --- Функции бота (без изменений) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам" (Factory Edition)</b>\n\n'
        'Отправьте фото — получите идеальную раскраску!\n\n'
        '<b>Команды:</b>\n'
        '• <code>/colors 24</code> — количество цветов (3-48)\n'
        '• <code>/detail 250</code> — мин. размер области (50-500)\n'
        '• <code>/help</code> — справка\n\n'
        '💡 Линии теперь плавные, а цифры не вылезают за края!',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Укажите число, например: <code>/colors 24</code>', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 48:
        await update.message.reply_text('❌ Допустимо от 3 до 48 цветов', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ Установлено {n_colors} цветов')


async def set_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Укажите число, например: <code>/detail 250</code>', parse_mode='HTML')
        return
    min_size = int(context.args[0])
    if not 50 <= min_size <= 500:
        await update.message.reply_text('❌ Допустимо от 50 до 500', parse_mode='HTML')
        return
    context.user_data['min_size'] = min_size
    await update.message.reply_text(f'✅ Мин. область: {min_size}px')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
    status_msg = await message.reply_text('🎨 Создаем фабричную раскраску...')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        min_size = context.user_data.get('min_size', MIN_REGION_SIZE)
        
        # Выносим тяжелую работу в отдельный поток, чтобы бот не зависал
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_size
        )
        
        await message.reply_photo(
            coloring_buffer,
            caption=f'🖼️ Раскраска готова!\n🎨 Цветов: {n_colors}\n📏 Отсечено мелких зон: < {min_size}px'
        )
        await message.reply_photo(palette_buffer, caption='🎨 Палитра для распечатки')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте фото попроще или измените /detail.')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка по "Factory Edition"</b>\n\n'
        '<b>Что изменилось:</b>\n'
        '• Линии стали гладкими (без зубцов)\n'
        '• Появились внутренние детали (глаза, пуговицы)\n'
        '• Цифры идеально вписаны в зоны\n'
        '• Убраны пиксельные артефакты\n\n'
        '<b>Команды:</b>\n'
        '• /colors N — цветов (3-48)\n'
        '• /detail N — мин. область (50-500)\n\n'
        '<b>Советы:</b>\n'
        '• Для детей: /colors 12 /detail 400\n'
        '• Для взрослых: /colors 32 /detail 150',
        parse_mode='HTML'
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('detail', set_detail))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 Factory Bot запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
