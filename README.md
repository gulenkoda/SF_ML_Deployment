# Credit Card Default ML Service

Итоговый учебный проект по дисциплине «Внедрение моделей машинного обучения».

В проекте реализован небольшой ML-сервис для прогнозирования дефолта по кредитной карте. Основная цель была не в том, чтобы получить максимально сложную модель, а в том, чтобы пройти полный минимальный цикл внедрения: подготовка данных, обучение модели, сохранение артефактов, Flask API, Docker, документация и план A/B-тестирования.

## Датасет

Используется датасет `Default of Credit Card Clients Dataset` из UCI Machine Learning Repository.

В проекте файл лежит здесь:

```text
data/UCI_Credit_Card.csv
```

Целевая переменная:

```text
default.payment.next.month
```

Значение `1` означает дефолт в следующем месяце, `0` — отсутствие дефолта.

## Структура проекта

```text
credit-default-ml-service/
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── model_loader.py
├── src/
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── model_v1.joblib
│   ├── model_v2.joblib
│   └── metrics.json
├── data/
│   └── UCI_Credit_Card.csv
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ARCHITECTURE.md
├── AB_TEST_PLAN.md
└── README.md
```

## Модели

В проекте используются две версии модели:

- `v1` — Logistic Regression с масштабированием признаков;
- `v2` — Random Forest Classifier.

Модель `v1` используется как базовая версия, а `v2` добавлена для демонстрации сравнения двух версий модели в рамках A/B-теста.

Метрики сохраняются в файл:

```text
models/metrics.json
```

Туда также записываются версия `scikit-learn`, версия Python, список признаков и `random_state`. Это сделано для воспроизводимости, потому что сохранённые через `joblib` модели могут зависеть от версии `scikit-learn`.

Текущий результат на тестовой выборке:

| Версия | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| v1 | 0.4613 | 0.3672 | 0.6202 | 0.7081 |
| v2 | 0.5398 | 0.4997 | 0.5870 | 0.7740 |

## Локальный запуск

Создать виртуальное окружение и установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Переобучить модели при необходимости:

```bash
python src/train.py
```

Запустить API локально:

```bash
python -m app.api
```

Проверить, что сервис работает:

```bash
curl http://localhost:5000/health
```

Ожидаемый ответ:

```json
{
  "available_models": ["v1", "v2"],
  "service": "credit-default-prediction",
  "status": "healthy"
}
```

## Пример запроса к `/predict`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "v1",
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
  }'
```

Пример ответа:

```json
{
  "model_version": "v1",
  "prediction": 1,
  "probability": 0.775479
}
```

## Формат API

### `GET /health`

Эндпоинт для проверки работоспособности сервиса.

Возвращает:

- статус сервиса;
- название сервиса;
- список доступных версий моделей.

### `POST /predict`

Эндпоинт для получения прогноза.

На вход принимает JSON с признаками клиента. Поле `model_version` необязательное. Если его не передать, используется модель `v1`.

Ответ содержит:

- `prediction` — прогноз класса (`0` или `1`);
- `probability` — вероятность дефолта;
- `model_version` — версия модели, которая использовалась для прогноза.

## Docker

Docker-образ собирается на базе `python:3.12-slim`. В образ включены код API, сохранённые модели и зависимости. Датасет и обучающие скрипты не нужны для работы контейнера, поэтому в runtime-образ они не копируются.

Сборка образа:

```bash
docker build -t credit-default-ml-service:latest .
```

Запуск контейнера:

```bash
docker run -p 5000:5000 credit-default-ml-service:latest
```

Если порт `5000` на компьютере уже занят, можно запустить контейнер на другом внешнем порту, например:

```bash
docker run -p 5001:5000 credit-default-ml-service:latest
```

Тогда проверка будет такой:

```bash
curl http://localhost:5001/health
```

## Docker Hub

Образ опубликован в Docker Hub:

```text
https://hub.docker.com/r/gulenkoda/credit-default-ml-service
```

Команды для публикации образа:

```bash
docker tag credit-default-ml-service:latest gulenkoda/credit-default-ml-service:latest
docker push gulenkoda/credit-default-ml-service:latest
```

## Docker Compose

Также можно запустить сервис через Docker Compose:

```bash
docker compose up --build
```

Файл `docker-compose.yml` запускает ML-сервис и пробрасывает порт `5000`.

## Логирование

API пишет технические логи запросов в файл:

```text
logs/api_requests.jsonl
```

Формат — JSON Lines. В реальной production-среде такие логи можно отправлять, например, в ELK-стек или другую систему мониторинга.

## ONNX-ML

В этом проекте модель сохраняется через `joblib`, потому что это простой вариант для Python-сервиса. В production модель можно дополнительно преобразовать в ONNX-ML через `skl2onnx` и запускать через ONNX Runtime. Это может быть полезно, если модель нужно использовать вне Python-среды или ускорить инференс.

## Тесты

Запуск тестов:

```bash
pytest
```

Тесты проверяют:

- работу `/health`;
- прогнозы для `v1` и `v2`;
- выбор модели по умолчанию;
- обработку некорректных входных данных;
- JSON-ответы для ошибок.

## Дополнительная документация

В проекте также есть отдельные файлы:

- `ARCHITECTURE.md` — описание архитектуры, монолитного подхода, логирования, RabbitMQ, DVC, MLflow и бизнес-метрик;
- `AB_TEST_PLAN.md` — план A/B-теста для сравнения моделей `v1` и `v2`.
