FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей отдельным слоем для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# В образ кладём только то, что нужно для инференса: код сервиса и модели.
# Датасет и обучающие скрипты в runtime-образ не попадают.
COPY app/ ./app/
COPY models/ ./models/

# Непривилегированный пользователь и каталог логов, доступный ему для записи
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Проверка работоспособности контейнера через эндпоинт /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').status==200 else 1)"

CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app.api:app"]
