# 1. Базовый образ
FROM python:3.12-slim

# 2. Создаем рабочую директорию (чтобы избежать проблем с /app)
WORKDIR /app

# 3. Устанавливаем системные пакеты ОДИН раз при сборке образа
RUN apt-get update && apt-get install -y curl

# 4. Устанавливаем Python-пакеты ОДИН раз при сборке образа
RUN pip install --no-cache-dir python-telegram-bot aiohttp httpx

# 5. Скачиваем код бота ОДИН раз при сборке образа
RUN curl -sL https://raw.githubusercontent.com/sheiten/mybot/main/bot.py -o /app/bot.py

# 6. Указываем команду, которая будет работать в foreground при старте контейнера
CMD ["python", "/app/bot.py"]
