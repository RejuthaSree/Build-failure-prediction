"""Single-record and batch inference for build-failure risk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from . import config
from .train import load_model

#: Probability thresholds used to bucket a build into a human-friendly risk level.
RISK_THRESHOLDS = {"low": 0.35, "medium": 0.65}


def risk_level(probability: float) -> str:
    """Translate a failure probability into ``low`` / ``medium`` / ``high``."""
    if probability < RISK_THRESHOLDS["low"]:
        return "low"
    if probability < RISK_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def validate_record(record: dict) -> dict:
    """Ensure every model feature is present; ignore unknown keys."""
    missing = [column for column in config.FEATURE_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    return {column: record[column] for column in config.FEATURE_COLUMNS}


def predict_build(model: Pipeline, record: dict) -> dict[str, object]:
    """Score a single build described by a feature dictionary.

    Returns a dictionary with the binary decision, the calibrated-looking
    probability and a risk bucket that a CI job can act on.
    """
    row = pd.DataFrame([validate_record(record)])
    probability = float(model.predict_proba(row)[0, 1])
    failed = bool(model.predict(row)[0])
    return {
        "build_failed": failed,
        "failure_probability": round(probability, 4),
        "risk_level": risk_level(probability),
    }


def predict_batch(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Score a whole dataframe of build records.

    Adds ``failure_probability``, ``predicted_failure`` and ``risk_level``
    columns alongside the original inputs.
    """
    missing = [column for column in config.FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing column(s): {missing}")

    features = frame[config.FEATURE_COLUMNS]
    probabilities = model.predict_proba(features)[:, 1]
    result = frame.copy()
    result["failure_probability"] = probabilities.round(4)
    result["predicted_failure"] = (probabilities >= 0.5).astype(int)
    result["risk_level"] = [risk_level(p) for p in probabilities]
    return result


def load_default_model(path: Path | str = config.MODEL_PATH) -> Pipeline:
    """Load the persisted model, raising a helpful error if it is absent."""
    return load_model(path)
