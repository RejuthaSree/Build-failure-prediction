# Build-Failure Prediction

A DevOps-focused machine-learning project that predicts whether a **CI/CD build
is likely to fail** *before* it runs, using commit and pipeline metadata
(e.g. lines changed, test coverage, prior failures, lint warnings).

The goal is to shift quality feedback left: a "build failure risk" score can be
used to gate pull requests, warn authors, or trigger extra checks.

## Why this matters in DevOps

Failed builds waste CI minutes, block merges and delay releases. A model that
flags high-risk changes lets teams:

- prioritise expensive pipeline stages only for risky commits,
- notify authors early, and
- track build-health trends over time.

## Project layout

```
Build-Failure Prediction/
├── src/build_failure/        # importable package (the pipeline)
│   ├── config.py             # paths, feature contract, constants
│   ├── data_generator.py     # synthetic CI build dataset
│   ├── preprocessing.py      # cleaning + feature/target split
│   ├── train.py              # model pipelines + cross-validated selection
│   ├── evaluate.py           # metrics + reporting
│   ├── predict.py            # single-record inference
│   └── api.py                # Flask REST API
├── tests/                    # pytest suite (one file per feature)
├── .github/workflows/        # GitHub Actions CI/CD
├── Dockerfile                # reproducible runtime
├── Makefile                  # one-liner task shortcuts
├── main.py                   # command-line entry point
└── docs/LEARNING.md          # assignment write-up (step by step)
```

## Quick start

```bash
python -m pip install -r requirements.txt

# 1. generate the synthetic dataset
python main.py generate-data --samples 5000

# 2. train and save the best model
python main.py train

# 3. evaluate on the held-out test set
python main.py evaluate

# 4. score a single build
python main.py predict --json '{"lines_added": 320, "test_coverage": 0.42}'

# 5. run the test suite
pytest
```

## Serving predictions

```bash
python -m build_failure.api      # starts http://127.0.0.1:5000
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"lines_added": 320, "test_coverage": 0.42}'
```

## Continuous integration

Every push runs the GitHub Actions workflow in
[.github/workflows/ci.yml](.github/workflows/ci.yml): it installs dependencies,
runs the full test suite on Python 3.10–3.12, and performs a training smoke test
on the synthetic data.

## Documentation

The step-by-step walkthrough of how this project was built — including a
description of **every commit and the feature it introduces** — is in
[docs/LEARNING.md](docs/LEARNING.md).
