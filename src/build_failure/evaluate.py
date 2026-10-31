"""Model evaluation.

Produces the headline classification metrics (accuracy, precision, recall,
F1, ROC-AUC) plus a confusion matrix, and can persist them as JSON so CI jobs
can archive a build-quality report.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config


def evaluate_model(pipeline: Pipeline, features, target) -> dict[str, object]:
    """Compute classification metrics for a fitted pipeline.

    ``zero_division=0`` keeps the metrics well defined when a tiny test set has
    no positive predictions, which otherwise raises a warning and returns NaN.
    """
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]
    matrix = confusion_matrix(target, predictions, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "n_samples": int(len(target)),
        "failure_rate": float(target.mean()),
    }


def format_report(metrics: dict[str, object]) -> str:
    """Render metrics as a human-readable block for the CLI."""
    matrix = metrics["confusion_matrix"]
    lines = [
        "Build-Failure Prediction - evaluation report",
        "=" * 46,
        f"samples evaluated : {metrics['n_samples']}",
        f"observed fail rate: {metrics['failure_rate']:.3f}",
        "-" * 46,
        f"accuracy          : {metrics['accuracy']:.4f}",
        f"precision         : {metrics['precision']:.4f}",
        f"recall            : {metrics['recall']:.4f}",
        f"f1 score          : {metrics['f1']:.4f}",
        f"roc-auc           : {metrics['roc_auc']:.4f}",
        "-" * 46,
        "confusion matrix (rows = actual, cols = predicted)",
        f"  TN={matrix['true_negative']:<6} FP={matrix['false_positive']}",
        f"  FN={matrix['false_negative']:<6} TP={matrix['true_positive']}",
    ]
    return "\n".join(lines)


def save_metrics(
    metrics: dict[str, object], path: Path | str = config.METRICS_PATH
) -> Path:
    """Write metrics to a JSON file, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return path
