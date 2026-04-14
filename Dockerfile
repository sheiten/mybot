FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir python-telegram-bot aiohttp httpx

# Вместо RUN curl используем COPY. 
# Docker сам увидит, если файл bot.py на гитхабе изменился, и обновит слой.
COPY bot.py .

CMD ["python", "bot.py"]
