import asyncio
import logging
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import aiohttp

# 🔧 НАСТРОЙКИ
import os
TOKEN = os.environ.get("BOT_TOKEN", "")
if not TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set!")

# Ссылки на актуальные списки прокси (обновляются каждые 4 часа)
URLS = {
    "ru": "https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_ru.txt",
    "eu": "https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_eu.txt",
    "all": "https://raw.githubusercontent.com/kort0881/telegram-proxy-collector/main/proxy_all.txt"
}

# Настройка логирования
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

async def fetch_proxies(url: str) -> list[str]:
    """Загружает список прокси из URL и возвращает список непустых строк."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    text = await response.text()
                    # Фильтруем пустые строки и строки без tg://
                    proxies = [line.strip() for line in text.splitlines() 
                               if line.strip().startswith("tg://")]
                    return proxies
                else:
                    logging.error(f"Ошибка загрузки {url}: статус {response.status}")
                    return []
    except Exception as e:
        logging.error(f"Исключение при загрузке {url}: {e}")
        return []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ответ на команду /start."""
    await update.message.reply_text(
        "🛡️ Бот для получения MTProto прокси с Fake TLS.\n\n"
        "Команды:\n"
        "/proxy [ru|eu|all] — одна случайная ссылка (по умолчанию ru)\n"
        "/list [N] [ru|eu|all] — список из N прокси (по умолчанию 3 ru)\n\n"
        "Источник: kort0881/telegram-proxy-collector"
    )

async def proxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выдаёт одну случайную прокси-ссылку."""
    # Разбираем аргументы
    region = "ru"
    if context.args:
        arg = context.args[0].lower()
        if arg in URLS:
            region = arg
        else:
            await update.message.reply_text(f"⚠️ Неизвестный регион '{arg}'. Доступно: ru, eu, all")
            return

    await update.message.reply_text(f"🔄 Загружаю список {region.upper()}...")
    proxies = await fetch_proxies(URLS[region])
    
    if not proxies:
        await update.message.reply_text("❌ Не удалось загрузить прокси. Попробуйте позже.")
        return

    chosen = random.choice(proxies)
    await update.message.reply_text(
        f"🎲 Случайный прокси ({region.upper()}):\n\n`{chosen}`",
        parse_mode="Markdown"
    )

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выдаёт список из нескольких прокси."""
    # Параметры по умолчанию
    count = 3
    region = "ru"

    # Парсим аргументы: первый может быть числом, второй — регионом
    args = context.args
    if args:
        # Проверяем, является ли первый аргумент числом
        try:
            count = int(args[0])
            if count < 1:
                count = 1
            elif count > 10:
                count = 10  # Ограничим, чтобы не спамить
            # Второй аргумент — регион
            if len(args) > 1 and args[1].lower() in URLS:
                region = args[1].lower()
        except ValueError:
            # Первый аргумент не число, значит это регион
            if args[0].lower() in URLS:
                region = args[0].lower()
            else:
                await update.message.reply_text(f"⚠️ Неизвестный параметр. Используйте: /list [число] [ru/eu/all]")
                return

    await update.message.reply_text(f"🔄 Загружаю {count} прокси из {region.upper()}...")
    proxies = await fetch_proxies(URLS[region])

    if not proxies:
        await update.message.reply_text("❌ Не удалось загрузить прокси.")
        return

    # Если запрошено больше, чем есть в списке
    if count > len(proxies):
        count = len(proxies)

    selected = random.sample(proxies, count)
    text = f"📋 **Прокси {region.upper()}** ({count} шт.):\n\n"
    for i, proxy in enumerate(selected, 1):
        text += f"{i}. `{proxy}`\n"

    await update.message.reply_text(text, parse_mode="Markdown")

def main() -> None:
    """Запуск бота."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("proxy", proxy_command))
    application.add_handler(CommandHandler("list", list_command))

    logging.info("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    main()
