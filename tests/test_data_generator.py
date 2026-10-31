"""Tests for the synthetic build dataset generator."""

import pandas as pd
import pytest

from build_failure import config
from build_failure.data_generator import generate_build_dataset, save_dataset


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_build_dataset(n_samples=1500, random_state=7)


def test_returns_dataframe_with_expected_schema(dataset: pd.DataFrame) -> None:
    assert isinstance(dataset, pd.DataFrame)
    assert list(dataset.columns) == config.FEATURE_COLUMNS + [config.TARGET_COLUMN]


def test_row_count_matches_request(dataset: pd.DataFrame) -> None:
    assert len(dataset) == 1500


def test_no_missing_values(dataset: pd.DataFrame) -> None:
    assert dataset.isna().sum().sum() == 0


def test_target_is_binary_and_has_both_classes(dataset: pd.DataFrame) -> None:
    target = dataset[config.TARGET_COLUMN]
    assert set(target.unique()).issubset({0, 1})
    # Both classes must appear, otherwise training is meaningless.
    assert target.nunique() == 2
    # Neither class should dominate completely.
    assert 0.1 < target.mean() < 0.9


def test_domain_constraints(dataset: pd.DataFrame) -> None:
    assert (dataset["test_coverage"] >= 0).all()
    assert (dataset["test_coverage"] <= 1).all()
    assert (dataset["files_changed"] >= 1).all()
    assert (dataset["commit_hour"] >= 0).all()
    assert (dataset["commit_hour"] <= 23).all()
    assert (dataset["is_weekend"].isin([0, 1])).all()


def test_generation_is_reproducible() -> None:
    a = generate_build_dataset(n_samples=300, random_state=123)
    b = generate_build_dataset(n_samples=300, random_state=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_data() -> None:
    a = generate_build_dataset(n_samples=300, random_state=1)
    b = generate_build_dataset(n_samples=300, random_state=2)
    assert not a.equals(b)


def test_invalid_sample_count_raises() -> None:
    with pytest.raises(ValueError):
        generate_build_dataset(n_samples=0)


def test_save_dataset_round_trips(tmp_path) -> None:
    frame = generate_build_dataset(n_samples=50, random_state=5)
    out = save_dataset(frame, tmp_path / "nested" / "builds.csv")
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert reloaded.shape == frame.shape
