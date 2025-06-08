# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Установим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем проект
COPY . .

# Укажем точку входа
CMD ["python", "src/model.py"]
