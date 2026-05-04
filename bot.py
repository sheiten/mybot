import asyncio
import logging
import os
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Для работы SLIC обязательно наличие модуля ximgproc (есть в стандартном pip install opencv-contrib-python)
try:
    import cv2.ximgproc
    SLIC_AVAILABLE = True
except ImportError:
    SLIC_AVAILABLE = False

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

MAX_IMAGE_SIZE = 1000
DEFAULT_REGION_SIZE = 1500 # Теперь это размер суперпикселей (площадь)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def process_image_for_coloring(photo_bytes: bytes, region_size: int = DEFAULT_REGION_SIZE):
    """Гибридный алгоритм: Сетка SLIC + Фабричные линии"""
    if not SLIC_AVAILABLE:
        raise RuntimeError("Установите opencv-contrib-python: pip install opencv-contrib-python")

    input_img = Image.open(io.BytesIO(photo_bytes)).convert('RGB')
    w, h = input_img.size
    
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        input_img = input_img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        w, h = input_img.size

    img_np = np.array(input_img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB) # SLIC лучше работает в пространстве LAB

    # 1. Инициализация SLIC (секрет аккуратных ячеек)
    # region_size регулирует размер "сот". Чем меньше число - тем мельче детали.
    slic = cv2.ximgproc.createSuperpixelSLIC(img_cv, region_size=region_size, ruler=20.0)
    slic.iterate(10) # 10 итераций достаточно для идеальной сходимости
    slic.enforceLabelConnectivity() # Убирает оторванные куски (важно!)

    labels = slic.getLabels() # Карта зон (каждый пиксель имеет номер зоны от 0 до N)
    num_superpixels = slic.getNumberOfSuperpixels()

    # 2. Собираем цвета зон и фильтруем мусор
    mask = slic.getLabelContourMask(thick_line=True) # Маска контуров (толстые линии)
    
    region_colors = {}
    region_areas = {}
    
    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            if label not in region_colors:
                region_colors[label] = [0, 0, 0]
                region_areas[label] = 0
            
            # Суммируем цвета в RGB (img_np в RGB)
            region_colors[label][0] += img_np[y, x, 0]
            region_colors[label][1] += img_np[y, x, 1]
            region_colors[label][2] += img_np[y, x, 2]
            region_areas[label] += 1

    # Усредняем цвета и отбрасываем слишком мелкие зоны (шум)
    valid_regions = {}
    for label, color_sum in region_colors.items():
        area = region_areas[label]
        if area < 300: # Пропускаем микроскопический мусор
            continue
        valid_regions[label] = [c // area for c in color_sum]

    # 3. Группировка похожих цветов (чтобы не было 500 разных номеров)
    # Собираем уникальные средние цвета
    unique_colors = list(valid_regions.values())
    
    if len(unique_colors) > 48:
        # Если зон слишком много, группируем их через KMeans ПАЛЕТРЫ
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=48, n_init=10, random_state=42)
        palette_labels = kmeans.fit_predict(unique_colors)
        palette_centers = kmeans.cluster_centers_.astype("uint8")
    else:
        # Если зон мало, просто нумеруем их
        palette_centers = np.array(unique_colors, dtype=np.uint8)
        palette_labels = list(range(len(unique_colors)))

    # Маппинг: оригинальная зона SLIC -> Номер цвета в палитре
    label_to_color_num = {}
    valid_idx = 0
    for label in valid_regions.keys():
        label_to_color_num[label] = palette_labels[valid_idx] + 1 # +1 чтобы было от 1 до 48
        valid_idx += 1

    # 4. Финальный рендеринг через PIL (для гладкости)
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except IOError:
        font = ImageFont.load_default()
        font_small = font

    # Вычисляем центры зон для цифр
    region_centers = {}
    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            if label not in valid_regions:
                continue
            if label not in region_centers:
                region_centers[label] = [0, 0, 0]
            region_centers[label][0] += x
            region_centers[label][1] += y
            region_centers[label][2] += 1

    # Рисуем цифры
    for label, center_data in region_centers.items():
        area = region_areas[label]
        cx = center_data[0] // area
        cy = center_data[1] // area
        
        num = label_to_color_num[label]
        label_str = str(num)
        current_font = font if area > 2000 else font_small
        
        bbox = draw.textbbox((0, 0), label_str, font=current_font)
        t_w = bbox[2] - bbox[0]
        t_h = bbox[3] - bbox[1]
        
        draw.text((cx - t_w / 2, cy - t_h / 2 - 2), label_str, fill=(100, 100, 100), font=current_font)

    # Накладываем ФАБРИЧНЫЕ КОНТУРЫ поверх всего
    canvas_np = np.array(canvas)
    # mask от SLIC: там где контур = 255 (белый). Инвертируем и красим в черный
    canvas_np[mask == 255] = [0, 0, 0]

    coloring_img = Image.fromarray(canvas_np)

    # 5. Генерация палитры
    palette = Image.new('RGB', (w, 80), (255, 255, 255))
    draw_pal = ImageDraw.Draw(palette)
    
    try:
        font_pal = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font_pal = ImageFont.load_default()

    swatch_w = w // len(palette_centers)
    for i, color in enumerate(palette_centers):
        x1, y1 = i * swatch_w + 4, 10
        x2, y2 = (i + 1) * swatch_w - 4, 50
        draw_pal.rectangle([x1, y1, x2, y2], fill=tuple(color.tolist()), outline=(180, 180, 180))
        
        label = str(i + 1)
        bbox = draw_pal.textbbox((0, 0), label, font=font_pal)
        t_w = bbox[2] - bbox[0]
        draw_pal.text((i * swatch_w + swatch_w//2 - t_w//2, 58), label, fill=(0, 0, 0), font=font_pal)

    # Сохранение в память
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG')
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette.save(palette_buffer, format='PNG')
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer, len(palette_centers)


# --- Бот ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not SLIC_AVAILABLE:
        await update.message.reply_text('❌ Ошибка: установите `opencv-contrib-python` на сервере.')
        return
        
    await update.message.reply_text(
        '🎨 <b>Суперпиксельный Бот (SLIC)</b>\n\n'
        'Теперь я использую алгоритм "пчелиных сот". Формы получаются аккуратными!\n\n'
        '⚙️ <b>Настройки:</b>\n'
        '• <code>/size 1500</code> — размер ячейки (500 = много мелких деталей, 3000 = крупные зоны)\n'
        '• <code>/help</code> — справка',
        parse_mode='HTML'
    )

async def set_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Укажите число: <code>/size 1500</code>', parse_mode='HTML')
        return
    size = int(context.args[0])
    if not 200 <= size <= 5000:
        await update.message.reply_text('❌ Допустимо от 200 до 5000', parse_mode='HTML')
        return
    context.user_data['region_size'] = size
    await update.message.reply_text(f'✅ Размер сот: {size}')

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    
    status_msg = await message.reply_text('🎨 Рисуем сетку суперпикселей...')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        region_size = context.user_data.get('region_size', DEFAULT_REGION_SIZE)
        
        coloring_buffer, palette_buffer, n_colors = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), region_size
        )
        
        await message.reply_photo(
            coloring_buffer,
            caption=f'🖼️ SLIC-Раскраска!\n🎨 Уникальных цветов в палитре: {n_colors}'
        )
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text(f'❌ Ошибка: {str(e)}')
    finally:
        await status_msg.delete()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Как работает SLIC:</b>\n\n'
        'Он режет картинку на одинаковые "соты", а потом подкрашивает их в цвета картинки.\n'
        'Это решает проблему рваных форм!\n\n'
        '<b>Команда /size:</b>\n'
        '• 500-800 — для мелких портретов (будет много цифр)\n'
        '• 1500 — стандартный баланс\n'
        '• 3000+ — для пейзажей (крупные зоны)',
        parse_mode='HTML'
    )

def main() -> None:
    if not SLIC_AVAILABLE:
        logger.error("Модуль cv2.ximgproc не найден! Установите opencv-contrib-python")
        return

    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('size', set_size))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 SLIC Bot запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
