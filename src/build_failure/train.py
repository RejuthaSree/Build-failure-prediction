"""Model training: candidate models, cross-validated selection and persistence.

The pipeline follows a simple, defensible recipe:

1. define a few candidate classifiers,
2. wrap each in the *same* preprocessing pipeline (feature scaling),
3. rank them with stratified cross-validated ROC-AUC,
4. refit the winner on the full training set and persist it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def candidate_models(random_state: int = config.RANDOM_STATE) -> dict[str, object]:
    """Return the classifiers we compare, cheap-to-expensive."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def build_pipeline(classifier: object) -> Pipeline:
    """Attach the shared preprocessing steps to a classifier.

    Scaling is a no-op for tree models but is essential for the logistic
    baseline, so using one pipeline for all candidates keeps the comparison fair.
    """
    return Pipeline(
        steps=[("scaler", StandardScaler()), ("classifier", classifier)]
    )


def cross_validate_models(
    features: pd.DataFrame,
    target: pd.Series,
    cv: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """Mean ROC-AUC from stratified CV for every candidate model."""
    scores: dict[str, float] = {}
    for name, classifier in candidate_models(random_state).items():
        pipeline = build_pipeline(classifier)
        fold_scores = cross_val_score(
            pipeline, features, target, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        scores[name] = float(np.mean(fold_scores))
    return scores


def select_best_model(scores: dict[str, float]) -> str:
    """Return the candidate name with the highest mean CV score."""
    if not scores:
        raise ValueError("scores must not be empty")
    return max(scores, key=scores.get)


def train_best_model(
    features: pd.DataFrame,
    target: pd.Series,
    cv: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> tuple[Pipeline, str, dict[str, float]]:
    """Cross-validate every candidate, then refit the best on all training data.

    Returns
    -------
    (pipeline, best_name, cv_scores)
    """
    scores = cross_validate_models(features, target, cv=cv, random_state=random_state)
    best_name = select_best_model(scores)
    best_pipeline = build_pipeline(candidate_models(random_state)[best_name])
    best_pipeline.fit(features, target)
    return best_pipeline, best_name, scores


def save_model(pipeline: Pipeline, path: Path | str = config.MODEL_PATH) -> Path:
    """Persist a fitted pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path | str = config.MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python main.py train` first."
        )
    return joblib.load(path)
