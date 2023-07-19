from fastapi.testclient import TestClient
from main import app
import pandas as pd
import pytest
from starter.ml.model import compute_model_metrics, inference
from starter.ml.data import process_data
import pickle

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


# ################# Test models #################
@pytest.fixture(scope="module", params=["model/rfmodel.pkl"])
def model(request):
    with open(request.param, "rb") as f:
        model = pickle.load(f)
    return model


@pytest.fixture(scope="module", params=["data/test-data.csv"])
def test_data(request):
    return pd.read_csv(request.param)


def test_inference(model, test_data):
    cat_features = test_data.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=model["encoder"],
        lb=model["lb"]
    )
    preds = inference(model=model["model"], X=X_test)
    assert len(preds) == X_test.shape[0]


def test_model_type(model):
    assert isinstance(model, dict)


def test_model_keys(model):
    assert set(model.keys()) == {"model", "encoder", "lb"}


def test_compute_model_metrics(model, test_data):
    cat_features = test_data.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=model["encoder"],
        lb=model["lb"]
    )
    preds = inference(model=model["model"], X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert precision >= 0 and isinstance(precision, float)
    assert recall >= 0 and isinstance(recall, float)
    assert fbeta >= 0 and isinstance(fbeta, float)