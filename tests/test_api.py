import pytest

from app.api import app

SAMPLE = {
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
    "PAY_AMT6": 0,
}


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "healthy"
    assert body["available_models"] == ["v1", "v2"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_predict_each_version(client, version):
    response = client.post("/predict", json={**SAMPLE, "model_version": version})
    body = response.get_json()
    assert response.status_code == 200
    assert body["model_version"] == version
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_default_version_is_v1(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.get_json()["model_version"] == "v1"


def test_predict_missing_field(client):
    bad = dict(SAMPLE)
    bad.pop("AGE")
    response = client.post("/predict", json=bad)
    assert response.status_code == 400
    assert "Missing fields" in response.get_json()["error"]


def test_predict_non_numeric_field(client):
    bad = dict(SAMPLE)
    bad["AGE"] = "twenty"
    response = client.post("/predict", json=bad)
    assert response.status_code == 400
    assert "numeric" in response.get_json()["error"]


def test_predict_unknown_version(client):
    response = client.post("/predict", json={**SAMPLE, "model_version": "v9"})
    assert response.status_code == 400
    assert "Unknown model version" in response.get_json()["error"]


def test_predict_malformed_json(client):
    response = client.post("/predict", data="{not valid", content_type="application/json")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_non_dict_body(client):
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400
    assert "object" in response.get_json()["error"]


def test_unknown_route_returns_json(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"


def test_method_not_allowed_returns_json(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.get_json()["error"] == "Method not allowed"
