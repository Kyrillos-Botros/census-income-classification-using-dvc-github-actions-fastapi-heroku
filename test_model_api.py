from fastapi.testclient import TestClient
from main import app
import pandas as pd
import json
import pytest
from starter.ml.model import compute_model_metrics, inference
from starter.ml.data import process_data
import pickle

client = TestClient(app)

@pytest.fixture(scope="module", params=["data/test-data.csv"])
def test_data(request):
    return pd.read_csv(request.param)

@pytest.fixture(scope="module", params=["model/rfmodel.pkl"])
def model(request):
    with open(request.param, "rb") as f:
        model = pickle.load(f)
    return model

@pytest.fixture()
def data_above_50k_func(test_data):
    data_above_50k = test_data[test_data["salary"] == ">50K"].copy().iloc[[3]]
    data_above_50k = data_above_50k.drop("salary", axis=1)
    data_above_50k_dict = data_above_50k.to_dict(orient="records")[0]
    return data_above_50k_dict

@pytest.fixture
def data_below_50k_func(test_data):
    data_below_50k = test_data[test_data["salary"] == "<=50K"].copy().iloc[:1]
    data_below_50k = data_below_50k.drop("salary", axis=1)
    data_below_50k_dict = data_below_50k.to_dict(orient="records")[0]
    return data_below_50k_dict

def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to census income classification APP"


def test_predict_above_50k(data_above_50k_func):

    response = client.post(
        "/predict",
        json=data_above_50k_func
    )
    result = set(response.json())

    assert response.status_code == 200
    assert 1 in result

def test_predict_below_50k(data_below_50k_func):


    response = client.post(
        "/predict",
        json=data_below_50k_func
    )
    result = set(response.json())
    assert response.status_code == 200
    assert 0 in result

def test_predict_type(test_data):
    response = client.post(
        "/predict",
        json=test_data.to_dict(orient="records")[0]
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_predict_length(test_data):
    response = client.post(
        "/predict",
       json= test_data.to_dict(orient="records")[0]
    )
    assert response.status_code == 200
    assert len(response.json()) == test_data.shape[0]


# ################# Test models #################

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
