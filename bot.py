import asyncio
import logging
import os
import io
import math
import ssl
import urllib.request
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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

# ============================================
# ФУНКЦИЯ ОБНОВЛЕНИЯ ЧЕРЕZ PORTAINER
# ============================================
def trigger_self_update():
    webhook_url = "https://2.26.116"  # Укажите ваш URL вебхука Portainer
    ctx = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(webhook_url, method='POST')
        with urllib.request.urlopen(req, context=ctx) as response:
            return response.getcode() == 204
    except Exception as e:
        logger.error(f"Ошибка обновления: {e}")
        return False

# ============================================
# ОСНОВНЫЕ ФУНКЦИИ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
# ============================================
def preprocess_image(image: Image.Image, target_size: int, strength: str = 'medium') -> np.ndarray:
    """Предобработка: изменение размера, медианное размытие"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    if strength == 'light':
        img_array = cv2.medianBlur(img_array, 5)
    elif strength == 'medium':
        img_array = cv2.medianBlur(img_array, 7)
    else:
        img_array = cv2.medianBlur(img_array, 11)
    
    return img_array

def cluster_colors_rgb(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Квантование цветов KMeans"""
    h, w = img_array.shape[:2]
    pixels = img_array.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    
    brightness = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    sorted_indices = np.argsort(brightness)
    centers = centers[sorted_indices]
    
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels = np.array([label_map[l] for l in labels]).reshape((h, w))
    
    quantized = centers[labels]
    return quantized, labels, [tuple(c) for c in centers]

def merge_with_morphology(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                          min_region_size: int) -> np.ndarray:
    """Объединяет мелкие регионы в крупные с помощью морфологии OpenCV"""
    h, w = quantized.shape[:2]
    new_quantized = quantized.copy()
    palette_np = np.array(palette)
    
    for color_idx, color in enumerate(palette):
        color_np = np.array(color)
        mask = np.all(quantized == color_np, axis=2).astype(np.uint8) * 255
        
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_region_size * 0.5:
                mask[labels_img == i] = 0
        
        new_quantized[mask > 0] = color_np
    
    return new_quantized

def create_coloring_page_raster(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                               min_region_size: int, line_thickness: int, 
                               line_color: str, font_size: int) -> io.BytesIO:
    """Создаёт растровую раскраску PNG с контурами и номерами"""
    h, w = quantized.shape[:2]
    
    # Цвет линий
    color_map = {
        'gray': [180, 180, 180],
        'dark': [100, 100, 100],
        'light': [210, 210, 210]
    }
    line_rgb = color_map.get(line_color, [180, 180, 180])
    
    # Создаём белый холст
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Для каждого цвета находим контуры
    for color_idx, color in enumerate(palette):
        color_np = np.array(color)
        mask = np.all(quantized == color_np, axis=2).astype(np.uint8) * 255
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_size:
                continue
            
            # Рисуем контур на холсте
            cv2.drawContours(canvas, [contour], -1, line_rgb, line_thickness)
    
    # Преобразуем в PIL для рисования номеров
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    placed_positions = []
    
    for color_idx, color in enumerate(palette):
        color_np = np.array(color)
        mask = np.all(quantized == color_np, axis=2).astype(np.uint8) * 255
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_size:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            too_close = False
            for px, py in placed_positions:
                if math.sqrt((cx - px) ** 2 + (cy - py) ** 2) < font_size * 3:
                    too_close = True
                    break
            if too_close:
                continue
            
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
    
    output = io.BytesIO()
    pil_canvas.save(output, format='PNG', dpi=(300, 300))
    output.seek(0)
    return output
def create_palette_image(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """Палитра"""
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
    """Основная функция обработки"""
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image, max_image_size, preprocess_strength)
    
    quantized, labels, palette = cluster_colors_rgb(img_array, n_colors)
    quantized = merge_with_morphology(quantized, palette, min_region_size)
    
    coloring_buffer = create_coloring_page_vector(
        quantized, palette, min_region_size, line_thickness, line_color, font_size
    )
    
    palette_buffer = io.BytesIO()
    palette_img = create_palette_image(palette)
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer

# ============================================
# ФУНКЦИИ БОТА
# ============================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам" (v2.1 Professional)</b>\n\n'
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
        '• <code>/help</code> — подробная справка\n'
        '• <code>/myid</code> — узнать свой ID (для админа)\n'
        '• <code>/update</code> — перезапустить бота (только для админа)',
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

# ============================================
# НОВЫЕ КОМАНДЫ
# ============================================

# ВАЖНО: Замените 1234567890 на ваш реальный Telegram ID
# Чтобы узнать свой ID, отправьте боту /myid
ADMIN_ID = 931848809  # <--- ЗАМЕНИТЕ НА ВАШ ID

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.message.reply_text(f'🆔 Ваш Telegram ID: <code>{user_id}</code>', parse_mode='HTML')

async def update_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id != ADMIN_ID:
        await update.message.reply_text('❌ У вас нет прав для этой команды.')
        return
    
    await update.message.reply_text('🚀 Запускаю обновление стека...')
    if trigger_self_update():
        await update.message.reply_text('✅ Portainer принял запрос. Контейнер будет перезапущен.')
    else:
        await update.message.reply_text('❌ Ошибка при отправке запроса в Portainer.')

# ============================================
# ОБРАБОТКА ФОТО
# ============================================

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
    
    status_msg = await message.reply_text(f'🎨 Создаю раскраску...\n\n{settings_text}')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors, min_region_size,
            line_thickness, line_color, font_size, max_image_size, preprocess_strength
        )
        
        await message.reply_document(
            document=coloring_buffer,
            filename='coloring_page.svg',
            caption=f'🖼️ Раскраска (SVG)\n{settings_text}'
        )
        await message.reply_photo(palette_buffer, caption='🎨 Палитра')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте другие настройки.')
    finally:
        await status_msg.delete()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Полная справка (v2.1)</b>\n\n'
        '<b>Основные команды:</b>\n'
        '• <code>/colors N</code> — цветов (3-48, по умолч. 24)\n'
        '• <code>/detail N</code> — мин. область (50-500, по умолч. 150)\n\n'
        '<b>Тонкая настройка:</b>\n'
        '• <code>/line N</code> — толщина линий (1-3, по умолч. 1)\n'
        '• <code>/linecolor X</code> — цвет линий (gray/dark/light)\n'
        '• <code>/font N</code> — размер шрифта (9-14, по умолч. 11)\n'
        '• <code>/size N</code> — макс. размер (800-3000, по умолч. 1500)\n'
        '• <code>/smooth X</code> — сглаживание (light/medium/strong)\n'
        '• <code>/settings</code> — показать настройки\n'
        '• <code>/myid</code> — узнать свой ID (для админа)\n'
        '• <code>/update</code> — перезапустить бота (только для админа)',
        parse_mode='HTML'
    )

# ============================================
# MAIN
# ============================================

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
    application.add_handler(CommandHandler('myid', myid))
    application.add_handler(CommandHandler('update', update_bot))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 Бот запущен (v2.1 Professional)!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
