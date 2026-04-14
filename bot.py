import asyncio
import logging
import random
import os
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import aiohttp

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

URLS = {
    'ru': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_ru.txt',
    'eu': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_eu.txt',
    'all': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_all.txt'
}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def fetch_proxies(url: str) -> list:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    text = await response.text()
                    proxies = [line.strip() for line in text.splitlines() if line.strip().startswith('tg://')]
                    return proxies
                else:
                    logging.error(f'Failed to load {url}: status {response.status}')
                    return []
    except Exception as e:
        logging.error(f'Exception loading {url}: {e}')
        return []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🛡️ Бот для получения MTProto прокси с Fake TLS.\n\n'
        'Команды:\n'
        '/proxy [ru|eu|all] — одна случайная ссылка\n'
        '/list [N] [ru|eu|all] — список из N прокси\n\n'
        'Источник: kort0881/telegram-proxy-collector'
    )

async def proxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    region = 'ru'
    if context.args and context.args[0].lower() in URLS:
        region = context.args[0].lower()
        
    await update.message.reply_text(f'🔄 Загружаю {region.upper()}...')
    proxies = await fetch_proxies(URLS[region])
    if not proxies:
        await update.message.reply_text('❌ Не удалось загрузить прокси.')
        return
    
    chosen = random.choice(proxies)
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Делаем ссылку кликабельной с помощью HTML-тега <a>
    clickable_link = f'<a href="{chosen}">🔗 Подключить прокси</a>'
    await update.message.reply_text(clickable_link, parse_mode='HTML')


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    count, region = 3, 'ru'
    args = context.args if context.args else []
    
    if args:
        try:
            count = int(args[0])
            count = max(1, min(count, 10))
            if len(args) > 1 and args[1].lower() in URLS:
                region = args[1].lower()
        except ValueError:
            if args[0].lower() in URLS:
                region = args[0].lower()
                
    await update.message.reply_text(f'🔄 Загружаю {count} прокси из {region.upper()}...')
    proxies = await fetch_proxies(URLS[region])
    if not proxies:
        await update.message.reply_text('❌ Не удалось загрузить прокси.')
        return
        
    if count > len(proxies):
        count = len(proxies)
        
    selected = random.sample(proxies, count)
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Каждая ссылка в списке тоже оборачивается в <a>
    text = '\n'.join([f'{i+1}. <a href="{p}">🔗 Прокси {i+1}</a>' for i, p in enumerate(selected)])
    await update.message.reply_text(text, parse_mode='HTML', disable_web_page_preview=True)


def main() -> None:
    PROXY_URL = os.environ.get('PROXY_URL', '') 
    
    builder = Application.builder().token(TOKEN)
    
    if PROXY_URL:
        try:
            # ИСПРАВЛЕНА ОШИБКА: раньше здесь было builder = builder.request(...), что ломало бота
            builder = builder.request(httpx.AsyncClient(proxy=PROXY_URL))
            logging.info(f"Бот использует прокси: {PROXY_URL}")
        except Exception as e:
            logging.error(f"Ошибка настройки прокси: {e}")

    application = builder.build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('proxy', proxy_command))
    application.add_handler(CommandHandler('list', list_command))
    logging.info('Бот запущен...')
    application.run_polling()

if __name__ == '__main__':
    main()
