# test update.
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

# ИСПРАВЛЕНО: Реальные ссылки из репозитория kort0881
URLS = {
    'all': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_all.txt',
    'ru': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_ru.txt',
    'eu': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_eu.txt',
    'verified': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_list.txt',
    'v': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_list.txt'
}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def fetch_proxies(url: str) -> list:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    text = await response.text()
                    # Фильтруем только валидные ссылки tg:// и пустые строки
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
        '🛡️ <b>Бот для получения MTProto прокси</b>\n\n'
        'Я беру proxies из актуального сбора kort0881.\n\n'
        '⚙️ <b>Доступные списки:</b>\n'
        '• <code>all</code> — Все прокси (микс, самое большое кол-во)\n'
        '• <code>ru</code> — Под РФ (Fake-TLS под Яндекс, VK и т.д.)\n'
        '• <code>eu</code> — Евросегмент (под Google, Amazon)\n'
        '• <code>verified</code> — Базовый проверенный список\n\n'
        '📝 <b>Как использовать:</b>\n'
        '/proxy — выдать 1 случайный из списка ALL\n'
        '/proxy ru — выдать 1 случайный из RU\n'
        '/list 5 — выдать список из 5 (из ALL)\n'
        '/list 3 eu — выдать список из 3 (из EU)\n\n'
        '💡 Можно писать просто тип без команды, например: <code>ru</code>',
        parse_mode='HTML'
    )

async def proxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # По умолчанию берем из 'all', так как там больше всего
    p_type = 'all' 
    if context.args:
        arg = context.args[0].lower()
        if arg in URLS:
            p_type = arg
        
    await update.message.reply_text(f'🔄 Загружаю {p_type.upper()} прокси...')
    proxies = await fetch_proxies(URLS[p_type])
    
    if not proxies:
        await update.message.reply_text(f'❌ Не удалось загрузить прокси ({p_type.upper()}). Список может быть временно пуст.')
        return
    
    chosen = random.choice(proxies)
    clickable_link = f'<a href="{chosen}">🔗 Подключить {p_type.upper()} прокси</a>'
    await update.message.reply_text(clickable_link, parse_mode='HTML')

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    count = 3
    p_type = 'all'
    
    args = context.args if context.args else []
    for arg in args:
        if arg.lower() in URLS:
            p_type = arg.lower()
        else:
            try:
                count = int(arg)
                count = max(1, min(count, 10)) # Ограничиваем до 10
            except ValueError:
                pass
                
    await update.message.reply_text(f'🔄 Загружаю {count} прокси ({p_type.upper()})...')
    proxies = await fetch_proxies(URLS[p_type])
    
    if not proxies:
        await update.message.reply_text(f'❌ Не удалось загрузить прокси.')
        return
        
    if count > len(proxies):
        count = len(proxies)
        
    selected = random.sample(proxies, count)
    
    text = '\n'.join([f'{i+1}. <a href="{p}">🔗 {p_type.upper()} Прокси {i+1}</a>' for i, p in enumerate(selected)])
    await update.message.reply_text(text, parse_mode='HTML', disable_web_page_preview=True)

# Эхо-обработчик
async def echo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.lower().strip().lstrip('/')
    
    if text in URLS:
        context.args = [text]
        await proxy_command(update, context)

def main() -> None:
    PROXY_URL = os.environ.get('PROXY_URL', '') 
    
    builder = Application.builder().token(TOKEN)
    
    if PROXY_URL:
        try:
            builder = builder.request(httpx.AsyncClient(proxy=PROXY_URL))
            logging.info(f"Бот использует прокси: {PROXY_URL}")
        except Exception as e:
            logging.error(f"Ошибка настройки прокси: {e}")

    application = builder.build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('proxy', proxy_command))
    application.add_handler(CommandHandler('list', list_command))
    
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_handler))
    
    logging.info('Бот запущен...')
    application.run_polling()

if __name__ == '__main__':
    main()
