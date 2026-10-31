"""Tests for data cleaning and train/test splitting."""

import numpy as np
import pandas as pd
import pytest

from build_failure import config
from build_failure.data_generator import generate_build_dataset, save_dataset
from build_failure.preprocessing import (
    clean_data,
    load_dataset,
    prepare_datasets,
    split_features_target,
    train_test_split_data,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_build_dataset(n_samples=800, random_state=11)


def test_clean_data_keeps_schema(raw: pd.DataFrame) -> None:
    cleaned = clean_data(raw)
    assert list(cleaned.columns) == config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_drops_duplicates_and_bad_rows() -> None:
    frame = pd.DataFrame(
        [
            {**{c: 1 for c in config.FEATURE_COLUMNS}, config.TARGET_COLUMN: 1},
            {**{c: 1 for c in config.FEATURE_COLUMNS}, config.TARGET_COLUMN: 1},  # dup
            {**{c: 2 for c in config.FEATURE_COLUMNS}, config.TARGET_COLUMN: 3},  # bad label
        ]
    )
    cleaned = clean_data(frame)
    assert len(cleaned) == 1


def test_clean_data_clips_out_of_range_values(raw: pd.DataFrame) -> None:
    dirty = raw.copy()
    dirty.loc[0, "test_coverage"] = 5.0
    dirty.loc[1, "commit_hour"] = 42
    cleaned = clean_data(dirty)
    assert cleaned["test_coverage"].max() <= 1.0
    assert cleaned["commit_hour"].max() <= 23


def test_clean_data_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"lines_added": [1, 2]}))


def test_split_features_target_shapes(raw: pd.DataFrame) -> None:
    features, target = split_features_target(clean_data(raw))
    assert list(features.columns) == config.FEATURE_COLUMNS
    assert set(target.unique()).issubset({0, 1})
    assert len(features) == len(target)


def test_train_test_split_is_stratified(raw: pd.DataFrame) -> None:
    features, target = split_features_target(clean_data(raw))
    x_train, x_test, y_train, y_test = train_test_split_data(features, target)
    assert len(x_train) + len(x_test) == len(features)
    # Class balance must be preserved between the two splits.
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_no_rows_are_shared_between_splits(raw: pd.DataFrame) -> None:
    features, target = split_features_target(clean_data(raw))
    x_train, x_test, _, _ = train_test_split_data(features, target)
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_prepare_datasets_end_to_end(tmp_path) -> None:
    path = save_dataset(generate_build_dataset(400, random_state=3), tmp_path / "d.csv")
    x_train, x_test, y_train, y_test = prepare_datasets(path)
    assert x_train.shape[1] == len(config.FEATURE_COLUMNS)
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)
    assert np.isfinite(x_train.to_numpy()).all()


def test_load_dataset_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")
