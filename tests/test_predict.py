"""Tests for single-record and batch prediction."""

import numpy as np
import pandas as pd
import pytest

from build_failure import config
from build_failure.data_generator import generate_build_dataset
from build_failure.predict import (
    predict_batch,
    predict_build,
    risk_level,
    validate_record,
)
from build_failure.preprocessing import clean_data, split_features_target
from build_failure.train import train_best_model


@pytest.fixture(scope="module")
def model():
    frame = clean_data(generate_build_dataset(n_samples=600, random_state=44))
    features, target = split_features_target(frame)
    pipeline, _, _ = train_best_model(features, target, cv=3)
    return pipeline


@pytest.fixture
def sample_record() -> dict:
    return {column: 1 for column in config.FEATURE_COLUMNS}


def test_predict_build_returns_expected_fields(model, sample_record) -> None:
    result = predict_build(model, sample_record)
    assert set(result) == {"build_failed", "failure_probability", "risk_level"}
    assert isinstance(result["build_failed"], bool)
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_level"] in {"low", "medium", "high"}


def test_missing_feature_raises(model) -> None:
    with pytest.raises(ValueError):
        predict_build(model, {"lines_added": 10})


def test_validate_record_ignores_unknown_keys() -> None:
    record = {column: 1 for column in config.FEATURE_COLUMNS}
    record["not_a_feature"] = 999
    cleaned = validate_record(record)
    assert set(cleaned) == set(config.FEATURE_COLUMNS)


def test_risk_level_buckets() -> None:
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.9) == "high"


def test_batch_prediction_preserves_length(model) -> None:
    frame = generate_build_dataset(n_samples=25, random_state=9)[config.FEATURE_COLUMNS]
    scored = predict_batch(model, frame)
    assert len(scored) == 25
    assert {"failure_probability", "predicted_failure", "risk_level"} <= set(scored.columns)
    assert scored["predicted_failure"].isin([0, 1]).all()


def test_batch_prediction_requires_features(model) -> None:
    with pytest.raises(ValueError):
        predict_batch(model, pd.DataFrame({"lines_added": [1, 2]}))


def test_higher_coverage_does_not_increase_risk(model) -> None:
    """Coverage has a strong negative weight, so the risk should be monotone."""
    base = {
        "lines_added": 500,
        "lines_deleted": 200,
        "files_changed": 12,
        "prior_failures": 3,
        "dependencies_changed": 1,
        "commit_hour": 23,
        "is_weekend": 1,
        "distinct_authors": 4,
        "changed_test_files": 0,
        "avg_review_time_min": 5,
        "open_issues": 30,
        "test_count": 40,
        "lint_warnings": 20,
    }
    low_coverage = predict_build(model, {**base, "test_coverage": 0.05})
    high_coverage = predict_build(model, {**base, "test_coverage": 0.95})
    assert low_coverage["failure_probability"] >= high_coverage["failure_probability"]


def test_probabilities_are_stable(model, sample_record) -> None:
    first = predict_build(model, sample_record)["failure_probability"]
    second = predict_build(model, sample_record)["failure_probability"]
    assert np.isclose(first, second)
