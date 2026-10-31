"""Tests for model evaluation and metric reporting."""

import json

import numpy as np
import pytest

from build_failure.data_generator import generate_build_dataset
from build_failure.evaluate import evaluate_model, format_report, save_metrics
from build_failure.preprocessing import (
    clean_data,
    split_features_target,
    train_test_split_data,
)
from build_failure.train import train_best_model


class _PerfectClassifier:
    """Minimal stand-in that returns the true labels, to test metric bounds."""

    def predict(self, features):
        return np.asarray(features)[:, -1]

    def predict_proba(self, features):
        label = np.asarray(features)[:, -1].astype(float)
        return np.column_stack([1 - label, label])


@pytest.fixture(scope="module")
def trained_split():
    frame = clean_data(generate_build_dataset(n_samples=700, random_state=33))
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split_data(features, target)
    pipeline, _, _ = train_best_model(x_train, y_train, cv=3)
    return pipeline, x_test, y_test


def test_metric_keys_present(trained_split) -> None:
    pipeline, x_test, y_test = trained_split
    metrics = evaluate_model(pipeline, x_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
        assert key in metrics


def test_metric_values_in_valid_range(trained_split) -> None:
    pipeline, x_test, y_test = trained_split
    metrics = evaluate_model(pipeline, x_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
    # The model should beat random guessing on the held-out set.
    assert metrics["roc_auc"] > 0.75


def test_confusion_matrix_sums_to_sample_count(trained_split) -> None:
    pipeline, x_test, y_test = trained_split
    matrix = evaluate_model(pipeline, x_test, y_test)["confusion_matrix"]
    assert sum(matrix.values()) == len(y_test)


def test_perfect_predictor_scores_one() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    features = np.column_stack([np.zeros_like(labels), labels])
    metrics = evaluate_model(_PerfectClassifier(), features, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["false_positive"] == 0


def test_format_report_contains_headline_metrics(trained_split) -> None:
    pipeline, x_test, y_test = trained_split
    report = format_report(evaluate_model(pipeline, x_test, y_test))
    assert "roc-auc" in report
    assert "confusion matrix" in report
    assert "accuracy" in report


def test_save_metrics_writes_valid_json(trained_split, tmp_path) -> None:
    pipeline, x_test, y_test = trained_split
    metrics = evaluate_model(pipeline, x_test, y_test)
    path = save_metrics(metrics, tmp_path / "out" / "metrics.json")
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["n_samples"] == metrics["n_samples"]
