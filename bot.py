import asyncio
import logging
import os
import io
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

# Параметры обработки
DEFAULT_N_COLORS = 12
MIN_REGION_SIZE = 100
MAX_IMAGE_SIZE = 1000
FONT_SIZE = 14

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# === ОБРАБОТКА ИЗОБРАЖЕНИЙ ===

def preprocess_image(image: Image.Image, target_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    img_array = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    return img_array


def cluster_colors(img_array: np.ndarray, n_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    quantized = centers[labels].reshape(h, w, c)
    
    brightness = [np.mean(color) for color in centers]
    sorted_indices = np.argsort(brightness)
    centers_sorted = centers[sorted_indices]
    
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels_mapped = np.array([label_map[l] for l in labels]).reshape(h, w)
    
    return quantized.astype(np.uint8), [tuple(c) for c in centers_sorted], labels_mapped


def find_regions(labels: np.ndarray, min_size: int = MIN_REGION_SIZE) -> List[dict]:
    h, w = labels.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for y in range(h):
        for x in range(w):
            if not visited[y, x]:
                color = labels[y, x]
                region_pixels = []
                queue = [(y, x)]
                visited[y, x] = True
                
                while queue:
                    cy, cx = queue.pop(0)
                    region_pixels.append((cy, cx))
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and 
                            not visited[ny, nx] and labels[ny, nx] == color):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                
                if len(region_pixels) >= min_size:
                    ys, xs = zip(*region_pixels)
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    center = (np.mean(xs), np.mean(ys))
                    
                    contour_pixels = []
                    for py, px in region_pixels:
                        is_border = False
                        for dy, dx in directions:
                            ny, nx = py + dy, px + dx                            if (0 <= ny < h and 0 <= nx < w and labels[ny, nx] != color):
                                is_border = True
                                break
                        if is_border:
                            contour_pixels.append((px, py))
                    
                    regions.append({
                        'color_idx': int(color),
                        'pixels': region_pixels,
                        'contour': contour_pixels,
                        'center': center,
                        'bbox': bbox,
                        'size': len(region_pixels)
                    })
    
    regions.sort(key=lambda r: r['size'], reverse=True)
    return regions


def create_coloring_page(width: int, height: int, regions: List[dict], palette: List[Tuple[int, int, int]]) -> Tuple[Image.Image, Image.Image]:
    coloring = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(coloring)
    
    for region in regions:
        if region['contour']:
            for x, y in region['contour']:
                draw.rectangle([(x-1, y-1), (x+1, y+1)], fill='black', outline='black')
    
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    for idx, region in enumerate(regions, start=1):
        cx, cy = region['center']
        draw.text((cx - 5, cy - 7), str(idx), fill='black', font=font, anchor='mm')
    
    n_colors = len(palette)
    palette_img = Image.new('RGB', (200, 50 + n_colors * 40), 'white')
    palette_draw = ImageDraw.Draw(palette_img)
    palette_draw.text((10, 10), "Палитра:", fill='black', font=font)
    
    for idx, color in enumerate(palette, start=1):
        y_pos = 40 + (idx - 1) * 40
        palette_draw.rectangle([(10, y_pos), (40, y_pos + 30)], fill=color, outline='black')
        palette_draw.text((50, y_pos + 15), f"{idx}.", fill='black', font=font)
        r, g, b = color
        palette_draw.text((70, y_pos + 15), f"RGB({r},{g},{b})", fill='gray', font=font)
    
    return coloring, palette_img

def process_image_for_coloring(photo_bytes: bytes, n_colors: int = DEFAULT_N_COLORS) -> Tuple[io.BytesIO, io.BytesIO]:
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image)
    h, w = img_array.shape[:2]
    
    _, palette, labels = cluster_colors(img_array, n_colors)
    regions = find_regions(labels)
    coloring_img, palette_img = create_coloring_page(w, h, regions, palette)
    
    coloring_buffer = io.BytesIO()
    coloring_img.save(coloring_buffer, format='PNG')
    coloring_buffer.seek(0)
    
    palette_buffer = io.BytesIO()
    palette_img.save(palette_buffer, format='PNG')
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer


# === ОБРАБОТЧИКИ ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Бот "Раскраска по номерам"</b>\n\n'
        'Отправь мне любое изображение — я превращу его в раскраску!\n\n'
        '⚙️ <b>Команды:</b>\n'
        '• <code>/colors 8</code> — количество цветов (3-30)\n'
        '• <code>/minsize 50</code> — мин. размер области (20-500)\n'
        '• <code>/help</code> — справка\n\n'
        '💡 Лучше работают изображения с чёткими контурами!',
        parse_mode='HTML'
    )


async def set_colors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Использование: <code>/colors 8</code> (3-30)', parse_mode='HTML')
        return
    n_colors = int(context.args[0])
    if not 3 <= n_colors <= 30:
        await update.message.reply_text('❌ Число должно быть от 3 до 30', parse_mode='HTML')
        return
    context.user_data['n_colors'] = n_colors
    await update.message.reply_text(f'✅ Установлено {n_colors} цветов 🎨')


async def set_minsize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text('❌ Использование: <code>/minsize 50</code> (20-500)', parse_mode='HTML')
        return
    min_size = int(context.args[0])
    if not 20 <= min_size <= 500:
        await update.message.reply_text('❌ Число должно быть от 20 до 500', parse_mode='HTML')
        return
    context.user_data['min_size'] = min_size
    await update.message.reply_text(f'✅ Мин. размер области: {min_size} пикселей')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    status_msg = await message.reply_text('🔄 Обрабатываю... (~10-30 сек)')
    
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        n_colors = context.user_data.get('n_colors', DEFAULT_N_COLORS)
        
        coloring_buffer, palette_buffer = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), n_colors
        )
        
        await message.reply_photo(coloring_buffer, caption=f'🖼️ Ваша раскраска!\n🎨 Цветов: {n_colors}')
        await message.reply_photo(palette_buffer, caption='🎨 Палитра цветов')
        
    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)
        await message.reply_text('❌ Ошибка обработки. Попробуйте другое изображение.')
    finally:
        await status_msg.delete()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 <b>Справка</b>\n\n'
        '<b>Команды:</b>\n'
        '• /start — начать\n'
        '• /colors N — цветов в палитре (3-30)\n'
        '• /minsize N — мин. размер области (20-500)\n'
        '• /help — эта справка\n\n'
        '📤 Просто отправь фото — получишь раскраску!',
        parse_mode='HTML'
    )


# === ЗАПУСК ===
def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('minsize', set_minsize))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    logger.info('🎨 Бот запущен!')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
