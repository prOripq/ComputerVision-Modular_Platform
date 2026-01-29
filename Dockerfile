# Используем официальный образ PyTorch (в нем уже есть CUDA)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Устанавливаем рабочую папку
WORKDIR /app

# Устанавливаем системные библиотеки для работы OpenCV (GLib)
# Без этого cv2 выдаст ошибку "ImportError: libGL.so.1..."
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем список библиотек и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Открываем порт 5000 (наш веб-сайт)
EXPOSE 5000

# Запускаем приложение
CMD ["python", "web_app.py"]