from pathlib import Path
import json
import logging

import joblib
import sklearn

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_FILES = {
    "v1": MODELS_DIR / "model_v1.joblib",
    "v2": MODELS_DIR / "model_v2.joblib",
}

_model_cache = {}


def _check_sklearn_version() -> None:
    '''
    Сравниваем версию scikit-learn в текущем окружении с той, в которой
    обучались модели. Несовпадение версий — частая причина того, что pickle
    модели загружается, но даёт ошибку или некорректные результаты при инференсе.
    '''
    try:
        with METRICS_PATH.open(encoding="utf-8") as file:
            trained_version = json.load(file).get("metadata", {}).get("sklearn_version")
    except (FileNotFoundError, ValueError):
        trained_version = None

    if trained_version and trained_version != sklearn.__version__:
        logger.warning(
            "Версия scikit-learn окружения (%s) не совпадает с версией обучения (%s). "
            "Возможны ошибки при загрузке модели.",
            sklearn.__version__,
            trained_version,
        )


def load_model(version: str = "v1"):
    """Load model by version and keep it in memory."""
    if version not in MODEL_FILES:
        raise ValueError(f"Unknown model version: {version}")
    if version not in _model_cache:
        model_path = MODEL_FILES[version]
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        _check_sklearn_version()
        _model_cache[version] = joblib.load(model_path)
    return _model_cache[version]


def available_versions():
    return sorted(MODEL_FILES.keys())
