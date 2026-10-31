"""Smoke tests for the project scaffolding."""

from build_failure import __version__
from build_failure import config


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_feature_contract_is_consistent() -> None:
    # No duplicate feature names and the target is not listed as a feature.
    assert len(config.FEATURE_COLUMNS) == len(set(config.FEATURE_COLUMNS))
    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS
    assert len(config.FEATURE_COLUMNS) == 14


def test_ensure_directories_is_idempotent() -> None:
    config.ensure_directories()
    config.ensure_directories()
    assert config.MODEL_DIR.exists()
    assert config.REPORT_DIR.exists()
