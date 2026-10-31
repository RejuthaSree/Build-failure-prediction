"""Central configuration shared by every stage of the pipeline.

Keeping paths, the feature contract and reproducibility settings in one place
means the data generator, the trainer, the evaluator and the API all agree on
the same schema.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "raw" / "build_records.csv"
MODEL_PATH = MODEL_DIR / "build_failure_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------
DEFAULT_N_SAMPLES = 5000

#: Numeric input features fed to the model, in a fixed order.
FEATURE_COLUMNS: list[str] = [
    "lines_added",
    "lines_deleted",
    "files_changed",
    "test_coverage",
    "prior_failures",
    "dependencies_changed",
    "commit_hour",
    "is_weekend",
    "distinct_authors",
    "changed_test_files",
    "avg_review_time_min",
    "open_issues",
    "test_count",
    "lint_warnings",
]

#: Binary target: 1 == build failed, 0 == build succeeded.
TARGET_COLUMN = "build_failed"


def ensure_directories() -> None:
    """Create the directories the pipeline writes into (idempotent)."""
    for path in (RAW_DATA_PATH.parent, MODEL_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
