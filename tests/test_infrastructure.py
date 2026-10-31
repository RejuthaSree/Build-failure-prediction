"""Tests for the DevOps infrastructure files.

These guard the deployment contract: a container image, a Makefile and a CI
workflow must exist and keep the commands the rest of the project relies on.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
MAKEFILE = PROJECT_ROOT / "Makefile"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def test_dockerfile_exists_and_exposes_service() -> None:
    assert DOCKERFILE.exists()
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in content
    assert "EXPOSE 5000" in content
    assert "HEALTHCHECK" in content
    assert "build_failure.api" in content


def test_dockerignore_excludes_generated_artefacts() -> None:
    content = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in ("data", "models", "reports", ".git"):
        assert entry in content


def test_makefile_declares_core_targets() -> None:
    assert MAKEFILE.exists()
    content = MAKEFILE.read_text(encoding="utf-8")
    for target in ("install", "data", "train", "evaluate", "test", "api", "clean"):
        assert f"{target}:" in content
    # Recipe lines must be tab-indented for make to accept them.
    assert "\n\tpython -m pytest -v" in content


def test_ci_workflow_defines_test_and_smoke_jobs() -> None:
    assert CI_WORKFLOW.exists()
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert "test" in workflow["jobs"]
    assert "train-smoke" in workflow["jobs"]
    steps = workflow["jobs"]["test"]["steps"]
    assert any("pytest" in str(step.get("run", "")) for step in steps)
