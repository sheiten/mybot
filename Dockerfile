FROM python:3.12-slim
WORKDIR /app
# Ничего не копируем, так как скрипт скачивается через curl
CMD ["python", "bot.py"]
