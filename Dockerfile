# Базовый образ
FROM python:3.12-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# 1. Устанавливаем curl только ОДИН РАЗ — при сборке образа
RUN apt-get update && apt-get install -y curl

# 2. Устанавливаем библиотеки Python только ОДИН РАЗ — при сборке
RUN pip install --no-cache-dir python-telegram-bot aiohttp httpx

# 3. Скачиваем ваш bot.py из GitHub только ОДИН РАЗ — при сборке
RUN curl -sL https://raw.githubusercontent.com/sheiten/mybot/main/bot.py -o /app/bot.py

# 4. Команда по умолчанию: просто запускаем бота (без скачиваний!)
CMD ["python", "/app/bot.py"]
