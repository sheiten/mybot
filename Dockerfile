# Используем легковесный образ Python версии 3.12
FROM python3.12-slim

# Устанавливаем рабочую директорию внутри будущего контейнера
WORKDIR app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта (включая сам bot.py)
COPY . .

# Команда, которая выполнится при старте контейнера
CMD [python, bot.py]