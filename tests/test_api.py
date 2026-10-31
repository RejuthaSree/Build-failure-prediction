"""Tests for the Flask inference API."""

import pytest

from build_failure import config
from build_failure.api import create_app
from build_failure.data_generator import generate_build_dataset, save_dataset
from build_failure.preprocessing import (
    clean_data,
    load_dataset,
    split_features_target,
    train_test_split_data,
)
from build_failure.train import save_model, train_best_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api_model")
    data_path = save_dataset(
        generate_build_dataset(n_samples=400, random_state=55), tmp / "data.csv"
    )
    frame = clean_data(load_dataset(data_path))
    features, target = split_features_target(frame)
    x_train, _, y_train, _ = train_test_split_data(features, target)
    pipeline, _, _ = train_best_model(x_train, y_train, cv=3)
    model_path = save_model(pipeline, tmp / "model.joblib")

    app = create_app(model_path)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def record() -> dict:
    return {column: 2 for column in config.FEATURE_COLUMNS}


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_single_build(client, record) -> None:
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"build_failed", "failure_probability", "risk_level"}


def test_predict_rejects_non_object_body(client) -> None:
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400


def test_predict_rejects_missing_features(client) -> None:
    response = client.post("/predict", json={"lines_added": 10})
    assert response.status_code == 422
    assert "Missing required feature" in response.get_json()["error"]


def test_predict_batch_endpoint(client, record) -> None:
    response = client.post("/predict/batch", json=[record, record])
    assert response.status_code == 200
    body = response.get_json()
    assert len(body) == 2
    assert "risk_level" in body[0]


def test_predict_batch_rejects_empty_list(client) -> None:
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400
