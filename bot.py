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

# ИЗМЕНЕНО: Добавлены ссылки на файлы по ТИПАМ прокси из репозитория
# Это реальные файлы из репозитория, в которых лежат рабочие прокси
URLS = {
    'fake': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_fake.txt',
    'mtg': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_mtg.txt',
    'mtproto': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_mtproto.txt',
    'all': 'https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_all.txt'
}

# Для удобства добавим алиасы (сокращения), которые понимает бот
ALIASES = {
    'f': 'fake', 'fake': 'fake', 'fakettl': 'fake', 'tls': 'fake',
    'm': 'mtg', 'mtg': 'mtg', 'g': 'mtg',
    'p': 'mtproto', 'mtproto': 'mtproto', 'no_tls': 'mtproto',
    'all': 'all', 'a': 'all', 'всё': 'all', 'все': 'all'
}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def fetch_proxies(url: str) -> list:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    text = await response.text()
                    # Фильтруем только валидные ссылки tg://
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
        '⚙️ <b>Доступные типы прокси:</b>\n'
        '• <code>fake</code> — Fake TLS (рекомендуется, самые стабильные)\n'
        '• <code>mtg</code> — MTProxy Go (отличная скорость)\n'
        '• <code>mtproto</code> — Классический MTProto (без обфускации)\n'
        '• <code>all</code> — Все прокси сразу\n\n'
        '📝 <b>Как использовать:</b>\n'
        '/proxy — выдать 1 случайный Fake TLS\n'
        '/proxy mtg — выдать 1 случайный MTG\n'
        '/list 5 — выдать список из 5 Fake TLS\n'
        '/list 3 mtproto — выдать список из 3 классических\n\n'
        '💡 Можно писать просто тип без команды, например: <code>mtg</code>',
        parse_mode='HTML'
    )

async def proxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ИЗМЕНЕНО: Умное определение типа
    p_type = 'fake'  # По умолчанию отдаем самый популярный тип
    if context.args:
        arg = context.args[0].lower()
        if arg in ALIASES:
            p_type = ALIASES[arg]
        
    await update.message.reply_text(f'🔄 Загружаю {p_type.upper()} прокси...')
    proxies = await fetch_proxies(URLS[p_type])
    
    if not proxies:
        await update.message.reply_text(f'❌ Не удалось загрузить прокси типа {p_type.upper()}. Возможно, список сейчас пуст.')
        return
    
    chosen = random.choice(proxies)
    clickable_link = f'<a href="{chosen}">🔗 Подключить {p_type.upper()} прокси</a>'
    await update.message.reply_text(clickable_link, parse_mode='HTML')

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ИЗМЕНЕНО: Теперь бот понимает и /list mtg, и /list 5 mtg, и /list 5
    count = 3
    p_type = 'fake'
    
    args = context.args if context.args else []
    for arg in args:
        if arg.lower() in ALIASES:
            p_type = ALIASES[arg.lower()]
        else:
            try:
                count = int(arg)
                count = max(1, min(count, 10)) # Ограничиваем до 10, чтобы не спамил
            except ValueError:
                pass
                
    await update.message.reply_text(f'🔄 Загружаю {count} прокси ({p_type.upper()})...')
    proxies = await fetch_proxies(URLS[p_type])
    
    if not proxies:
        await update.message.reply_text(f'❌ Не удалось загрузить прокси.')
        return
        
    # Если запросили больше, чем есть в файле
    if count > len(proxies):
        count = len(proxies)
        
    selected = random.sample(proxies, count)
    
    text = '\n'.join([f'{i+1}. <a href="{p}">🔗 {p_type.upper()} Прокси {i+1}</a>' for i, p in enumerate(selected)])
    await update.message.reply_text(text, parse_mode='HTML', disable_web_page_preview=True)

# ИЗМЕНЕНО: Добавлен эхо-обработчик для максимального удобства
async def echo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Если пользователь просто пишет текст (не команду), проверяем, не запрос ли это прокси."""
    text = update.message.text.lower().strip()
    
    # Убираем слэш на случай, если юзер напишет "/fake" вместо команды
    text = text.lstrip('/')
    
    if text in ALIASES:
        # Подменяем аргументы и вызываем логику команды /proxy
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
    
    # Добавляем обработчик простых текстовых сообщений
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_handler))
    
    logging.info('Бот запущен...')
    application.run_polling()

if __name__ == '__main__':
    main()
