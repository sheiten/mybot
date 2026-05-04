import asyncio
import logging
import os
import io
from typing import List, Tuple
import math
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

# Настройки "фабричного" вида
MAX_IMAGE_SIZE = 1000
MIN_REGION_AREA = 250  # Размер минимальной детали. Чем больше, тем проще рисовать.
NUM_COLORS = 24       # Оптимальное количество цветов

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Упрощение изображения (эффект пластилина)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Убираем шум
    img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    # 2. Группируем пиксели в цветовые пятна
    shifted = cv2.pyrMeanShiftFiltering(img_bgr, 20, 45)
    # 3. Сглаживаем края
    return cv2.medianBlur(shifted, 5)

def apply_kmeans(img_bgr, num_colors):
    """Квантование (подбор палитры)"""
    pixels = img_bgr.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")
    quantized = centers[labels].reshape(img_bgr.shape)
    return quantized, centers

def create_coloring_page(quantized, centers):
    """Создание контуров, удаление мусора и нумерация"""
    h, w = quantized.shape[:2]
    cleaned = quantized.copy()
    
    # 1. Region Merging: удаляем микро-островки
    mask_visited = np.zeros((h + 2, w + 2), np.uint8)
    for y in range(h):
        for x in range(w):
            if mask_visited[y + 1, x + 1] == 0:
                color = cleaned[y, x].tolist()
                _, rect, area, _ = cv2.floodFill(cleaned, mask_visited, (x, y), color, 
                                                (0,0,0), (0,0,0), flags=4 | (255 << 8))
                if area < MIN_REGION_AREA:
                    nx, ny = max(0, x-1), max(0, y-1)
                    neighbor_color = cleaned[ny, nx].tolist()
                    cv2.floodFill(cleaned, None, (x, y), neighbor_color)

    # 2. Отрисовка контуров
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cleaned, kernel)
    edges = cv2.absdiff(dilated, cleaned)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    _, binary_edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY_INV)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[binary_edges == 0] = (200, 200, 200) # Светло-серые границы

    # 3. Простановка цифр
    for i, color in enumerate(centers):
        color_mask = cv2.inRange(cleaned, color, color)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_REGION_AREA: continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0:
                    cv2.putText(canvas, str(i + 1), (cX - 7, cY + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    return canvas, cleaned

def create_palette_bar(centers, width):
    """Нижняя панель с образцами красок"""
    swatch_w = width // len(centers)
    bar = np.ones((80, width, 3), dtype=np.uint8) * 255
    for i, color in enumerate(centers):
        x_start = i * swatch_w
        cv2.rectangle(bar, (x_start + 4, 10), (x_start + swatch_w - 4, 50), color.tolist(), -1)
        cv2.rectangle(bar, (x_start + 4, 10), (x_start + swatch_w - 4, 50), (180, 180, 180), 1)
        cv2.putText(bar, str(i + 1), (x_start + swatch_w // 2 - 5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return bar

# --- Пример интеграции в твой хендлер ---
async def process_photo_handler(message: types.Message):
    # 1. Загрузка
    photo_io = io.BytesIO()
    await message.photo[-1].download(destination_file=photo_io)
    photo_io.seek(0)
    input_img = Image.open(photo_io)

    # 2. Обработка
    simplified = preprocess_image(input_img)
    quantized, centers = apply_kmeans(simplified, NUM_COLORS)
    canvas, _ = create_coloring_page(quantized, centers)
    palette = create_palette_bar(centers, canvas.shape[1])
    
    # 3. Сборка финала
    final_img_np = np.vstack([canvas, palette])
    final_img_rgb = cv2.cvtColor(final_img_np, cv2.COLOR_BGR2RGB)
    
    # 4. Отправка
    output_io = io.BytesIO()
    Image.fromarray(final_img_rgb).save(output_io, format='PNG')
    output_io.seek(0)
    await message.answer_document(types.InputFile(output_io, filename="pbn_schema.png"))

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
