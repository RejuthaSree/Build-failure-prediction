"""Data cleaning and train/test splitting.

This stage turns the raw CSV into a deterministic, leakage-free
``(X_train, X_test, y_train, y_test)`` split that the trainer consumes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_dataset(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read a build dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python main.py generate-data` first."
        )
    return pd.read_csv(path)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates/invalid rows and enforce domain bounds.

    Only the columns required by the model are kept, which guards against
    accidentally leaking extra information (e.g. an id column) into training.
    """
    required = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = frame[required].copy()
    cleaned = cleaned.drop_duplicates().dropna()
    cleaned = cleaned[cleaned[config.TARGET_COLUMN].isin([0, 1])]

    # Clip physically impossible values caused by upstream data errors.
    cleaned["test_coverage"] = cleaned["test_coverage"].clip(0.0, 1.0)
    cleaned["commit_hour"] = cleaned["commit_hour"].clip(0, 23)

    for column in config.FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna()

    cleaned = cleaned.reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning; check the input dataset.")
    return cleaned


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model inputs from the binary target."""
    features = frame[config.FEATURE_COLUMNS].copy()
    target = frame[config.TARGET_COLUMN].astype(int).copy()
    return features, target


def train_test_split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the failure/success ratio."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def prepare_datasets(
    path: Path | str = config.RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience wrapper: load -> clean -> split in one call."""
    frame = clean_data(load_dataset(path))
    features, target = split_features_target(frame)
    return train_test_split_data(features, target)
