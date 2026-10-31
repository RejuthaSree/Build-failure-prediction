"""Synthetic CI/CD build dataset generation.

Real build logs are rarely shareable, so this module fabricates a statistically
plausible dataset that behaves like real CI metadata. The target is *derived*
from the features through a noise-injected logistic model, which means a model
trained on it can genuinely learn signal instead of memorising noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance scaling used only to weight the latent model."""
    return (values - values.mean()) / (values.std() + 1e-9)


def generate_build_dataset(
    n_samples: int = config.DEFAULT_N_SAMPLES,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Create a synthetic build dataset.

    Each row represents one CI build attempt. Feature distributions are chosen
    to be plausible (right-skewed change sizes, bounded coverage, Poisson counts)
    and the binary ``build_failed`` label is sampled from a logistic function of
    the features plus a little label noise.

    Parameters
    ----------
    n_samples:
        Number of build records to generate.
    random_state:
        Seed for the pseudo-random generator; identical seeds reproduce the
        exact same table.

    Returns
    -------
    pandas.DataFrame
        Columns = ``config.FEATURE_COLUMNS`` + ``config.TARGET_COLUMN``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.default_rng(random_state)

    # --- Raw feature distributions -------------------------------------
    lines_added = rng.gamma(shape=2.0, scale=110.0, size=n_samples)
    lines_deleted = rng.gamma(shape=2.0, scale=55.0, size=n_samples)
    files_changed = rng.poisson(lam=6.0, size=n_samples) + 1
    test_coverage = np.clip(rng.normal(0.62, 0.18, size=n_samples), 0.03, 0.99)
    prior_failures = rng.poisson(lam=1.2, size=n_samples)
    dependencies_changed = rng.binomial(1, 0.25, size=n_samples)
    commit_hour = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.binomial(1, 2 / 7, size=n_samples)
    distinct_authors = rng.integers(1, 6, size=n_samples)
    changed_test_files = rng.poisson(lam=2.0, size=n_samples)
    avg_review_time_min = np.clip(rng.gamma(2.0, 40.0, size=n_samples), 1.0, None)
    open_issues = rng.poisson(lam=15.0, size=n_samples)
    test_count = rng.poisson(lam=80.0, size=n_samples)
    lint_warnings = rng.poisson(lam=8.0, size=n_samples)

    # --- Latent failure model ------------------------------------------
    # Positive weights push a build towards failure; negative weights push it
    # towards success. This encodes intuitive DevOps relationships.
    logit = (
        -0.35
        + 0.95 * _standardise(lines_added)        # bigger diffs are riskier
        + 0.55 * _standardise(lines_deleted)      # lots of deletion => churn
        + 0.75 * _standardise(files_changed)      # wide blast radius
        - 1.15 * _standardise(test_coverage)      # good coverage protects
        + 0.90 * _standardise(prior_failures)     # flaky history
        + 0.70 * dependencies_changed             # dependency churn
        + 0.20 * _standardise(commit_hour - 14.0)  # late-night commits riskier
        + 0.35 * is_weekend
        + 0.25 * _standardise(distinct_authors)   # more hands, more conflicts
        - 0.30 * _standardise(changed_test_files)  # touching tests adds safety
        - 0.20 * _standardise(avg_review_time_min)  # more review => safer
        + 0.30 * _standardise(open_issues)        # repo health
        - 0.35 * _standardise(test_count)         # larger suite catches more
        + 0.60 * _standardise(lint_warnings)      # code-smell signal
    )

    failure_probability = _sigmoid(logit)
    build_failed = (rng.random(n_samples) < failure_probability).astype(int)

    frame = pd.DataFrame(
        {
            "lines_added": lines_added.round(0).astype(int),
            "lines_deleted": lines_deleted.round(0).astype(int),
            "files_changed": files_changed.astype(int),
            "test_coverage": test_coverage.round(4),
            "prior_failures": prior_failures.astype(int),
            "dependencies_changed": dependencies_changed.astype(int),
            "commit_hour": commit_hour.astype(int),
            "is_weekend": is_weekend.astype(int),
            "distinct_authors": distinct_authors.astype(int),
            "changed_test_files": changed_test_files.astype(int),
            "avg_review_time_min": avg_review_time_min.round(2),
            "open_issues": open_issues.astype(int),
            "test_count": test_count.astype(int),
            "lint_warnings": lint_warnings.astype(int),
            config.TARGET_COLUMN: build_failed,
        }
    )
    return frame[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]


def save_dataset(frame: pd.DataFrame, path: Path | str = config.RAW_DATA_PATH) -> Path:
    """Persist a dataset to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
