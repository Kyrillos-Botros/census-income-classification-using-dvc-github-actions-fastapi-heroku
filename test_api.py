from fastapi.testclient import TestClient
from main import app
import pandas as pd

client = TestClient(app)


def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to census income classification APP"


def test_predict_output():
    response = client.post(
        "/predict",
        json={
            "path": "data/test-data.csv"
        }
    )
    result = set(response.json())
    assert response.status_code == 200
    assert result.issubset({0, 1})


def test_predict_type():
    response = client.post(
        "/predict",
        json={
            "path": "data/test-data.csv"
        }
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_predict_length():
    response = client.post(
        "/predict",
        json={
            "path": "data/test-data.csv"
        }
    )
    df = pd.read_csv("data/test-data.csv")
    assert response.status_code == 200
    assert len(response.json()) == df.shape[0]
