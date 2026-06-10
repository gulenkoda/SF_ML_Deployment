from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = "default.payment.next.month"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(BASE_DIR / "data" / "UCI_Credit_Card.csv")
    X = df.drop(columns=[TARGET, "ID"])
    y = df[TARGET].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    results = {}
    for version in ["v1", "v2"]:
        model = joblib.load(BASE_DIR / "models" / f"model_{version}.joblib")
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[version] = {
            "classification_report": classification_report(y_test, pred, output_dict=True),
            "roc_auc": roc_auc_score(y_test, proba),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
