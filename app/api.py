from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from app.model_loader import available_versions, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "api_requests.jsonl"

try:
    LOG_DIR.mkdir(exist_ok=True)
except OSError as error:
    logger.warning("Could not create log directory %s: %s", LOG_DIR, error)

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def write_log(record: dict) -> None:
    '''
    Запись лога не должна влиять на ответ API. Если файл недоступен для
    записи (например, контейнер запущен под непривилегированным пользователем,
    а каталог логов смонтирован только для чтения), запрос всё равно
    обрабатывается, а ошибка логирования фиксируется отдельно.
    '''
    try:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as error:
        logger.warning("Failed to write request log: %s", error)


def preprocess_input(data: dict) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    values = []
    for col in FEATURE_COLUMNS:
        try:
            values.append(float(data[col]))
        except (TypeError, ValueError):
            raise ValueError(f"Field {col} must be numeric")
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "credit-default-prediction",
        "available_models": available_versions(),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    started_at = datetime.now(timezone.utc).isoformat()
    request_id = request.headers.get("X-Request-ID", "manual-request")
    model_version = "v1"

    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            raise ValueError("Request body must be valid JSON")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        model_version = payload.pop("model_version", "v1")
        if not isinstance(model_version, str):
            raise ValueError("model_version must be a string")

        model = load_model(model_version)
        features = preprocess_input(payload)

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        response = {
            "prediction": prediction,
            "probability": round(probability, 6),
            "model_version": model_version,
        }
        status_code = 200
    except ValueError as error:
        '''Некорректный ввод клиента: невалидный JSON, неизвестная версия модели, отсутствующие или нечисловые поля.'''
        response = {"error": str(error)}
        status_code = 400
    except Exception:
        '''Внутренний сбой сервиса (например, повреждённый или отсутствующий артефакт модели). Детали не возвращаем клиенту, но пишем в лог.'''
        logger.exception("Unhandled error in /predict")
        response = {"error": "Internal server error"}
        status_code = 500

    write_log({
        "timestamp": started_at,
        "request_id": request_id,
        "path": request.path,
        "model_version": model_version,
        "status_code": status_code,
        "response": response,
    })
    return jsonify(response), status_code


@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def handle_method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def handle_server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
