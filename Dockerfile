FROM python:3.12-slim

WORKDIR /app

# Системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем ВЕСЬ код проекта (включая bot.py, генераторы и т.д.)
COPY . .

# Запуск
CMD ["python", "bot.py"]
