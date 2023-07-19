import pytest
from starter.ml.model import compute_model_metrics, inference
from starter.ml.data import process_data
import pickle
import pandas as pd


@pytest.fixture(scope="module", params=["../model/rfmodel.pkl"])
def model(request):
    with open(request.param, "rb") as f:
        model = pickle.load(f)
    return model


@pytest.fixture(scope="module", params=["../data/test-data.csv"])
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
