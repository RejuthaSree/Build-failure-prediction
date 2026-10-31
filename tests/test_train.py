"""Tests for the model training pipeline."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from build_failure import config
from build_failure.data_generator import generate_build_dataset
from build_failure.preprocessing import (
    clean_data,
    split_features_target,
    train_test_split_data,
)
from build_failure.train import (
    build_pipeline,
    candidate_models,
    cross_validate_models,
    load_model,
    save_model,
    select_best_model,
    train_best_model,
)


@pytest.fixture(scope="module")
def split():
    frame = clean_data(generate_build_dataset(n_samples=800, random_state=21))
    features, target = split_features_target(frame)
    return train_test_split_data(features, target)


def test_candidates_are_distinct_models() -> None:
    models = candidate_models()
    assert set(models) == {"logistic_regression", "random_forest", "gradient_boosting"}


def test_build_pipeline_has_scaler_and_classifier() -> None:
    pipeline = build_pipeline(candidate_models()["random_forest"])
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]


def test_cross_validation_scores_are_probabilities(split) -> None:
    x_train, _, y_train, _ = split
    scores = cross_validate_models(x_train, y_train, cv=3)
    assert set(scores) == set(candidate_models())
    for value in scores.values():
        assert 0.5 <= value <= 1.0


def test_select_best_model_returns_argmax() -> None:
    assert select_best_model({"a": 0.7, "b": 0.9, "c": 0.8}) == "b"
    with pytest.raises(ValueError):
        select_best_model({})


def test_train_best_model_learns_signal(split) -> None:
    x_train, _, y_train, _ = split
    pipeline, best_name, scores = train_best_model(x_train, y_train, cv=3)
    # The synthetic data is learnable, so ROC-AUC must be clearly better
    # than random guessing.
    assert scores[best_name] > 0.75
    assert hasattr(pipeline, "predict_proba")
    predictions = pipeline.predict(x_train)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_save_and_load_model_round_trip(split, tmp_path) -> None:
    x_train, _, y_train, _ = split
    pipeline, _, _ = train_best_model(x_train, y_train, cv=3)
    path = save_model(pipeline, tmp_path / "model.joblib")
    assert path.exists()
    restored = load_model(path)
    np.testing.assert_allclose(
        pipeline.predict_proba(x_train), restored.predict_proba(x_train)
    )


def test_load_model_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")
