from pathlib import Path
import json
import platform
from datetime import datetime, timezone

import joblib
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "UCI_Credit_Card.csv"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
TARGET = "default.payment.next.month"
DROP_COLUMNS = ["ID"]
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET].astype(int)
    return X, y


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "v1": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "v2": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    metrics = {}
    for version, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"model_{version}.joblib")
        metrics[version] = evaluate_model(model, X_test, y_test)

    '''
    Версии библиотек и список признаков сохраняются вместе с метриками.
    Это нужно, чтобы артефакты модели и окружение, в котором они получены,
    нельзя было незаметно рассинхронизировать. При загрузке модели сервис
    сверяет фактическую версию scikit-learn с этим значением.
    '''
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_columns": list(X.columns),
        "target": TARGET,
    }

    payload = {"metadata": metadata, "metrics": metrics}
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
