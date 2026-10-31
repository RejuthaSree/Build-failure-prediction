# Build-Failure Prediction — Assignment Documentation & Learnings

**Repository:** <https://github.com/RejuthaSree/Build-failure-prediction>
**Domain:** DevOps / CI-CD quality engineering
**Tech stack:** Python 3.11, scikit-learn, pandas, NumPy, Flask, pytest, GitHub Actions, Docker
**Author:** Rejutha Sree

---

## 1. Objective

Failed CI/CD builds waste pipeline minutes, block merges and delay releases. The
goal of this assignment is to build a **Build-Failure Prediction** system that
uses machine learning to predict, *before a pipeline runs*, whether a build is
likely to fail — based on commit and pipeline metadata such as:

| Signal | Why it matters |
| --- | --- |
| `lines_added` / `lines_deleted` / `files_changed` | Large or wide-reaching changes are riskier |
| `test_coverage` | Good coverage catches regressions before CI |
| `prior_failures` | Flaky history predicts future failure |
| `dependencies_changed` | Dependency churn frequently breaks builds |
| `commit_hour` / `is_weekend` | Off-hours commits correlate with higher failure |
| `changed_test_files`, `test_count` | A healthy, growing test suite is protective |
| `lint_warnings`, `open_issues` | Code-smell and repo-health signals |

The deliverable is an end-to-end, reproducible MLOps-style project: data
generation → preprocessing → training → evaluation → serving (CLI + REST API) →
CI/CD automation → containerisation, plus this document.

---

## 2. Development method

The project was built **incrementally, one feature per commit**:

1. Implement exactly one feature.
2. Write tests for that feature.
3. Run the full test suite.
4. **Only after the tests pass**, create a single descriptive commit.

This is why the Git history reads like a checklist of features, and why every
commit below is independently understandable and revertible.

### Commit map

| # | Commit | Type | Feature introduced | Tests after commit |
| --- | --- | --- | --- | --- |
| 1 | `bb27e23` | chore | Project scaffolding, config, test harness | 3 |
| 2 | `ada9571` | feat | Synthetic CI build dataset generator | 12 |
| 3 | `4236159` | feat | Data cleaning + stratified train/test split | 21 |
| 4 | `4d33f47` | feat | Cross-validated training pipeline + persistence | 28 |
| 5 | `2e542d4` | feat | Evaluation metrics + JSON report | 34 |
| 6 | `ac126af` | feat | Prediction module + CLI | 47 |
| 7 | `4353500` | feat | Flask REST API | 53 |
| 8 | `ff1f74b` | ci | GitHub Actions CI/CD pipeline | 53 |
| 9 | `e809ffd` | build | Docker image + Makefile | 57 |
| 10 | *(this file)* | docs | Learning document | 57 |

Final state: **57 automated tests passing**.

---

## 3. Environment setup

| Tool | Version used |
| --- | --- |
| Python | 3.11.9 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| NumPy | 2.4.6 |
| Flask | 3.1.2 |
| pytest | 9.1.1 |
| Git | 2.53.0 |

```bash
git clone https://github.com/RejuthaSree/Build-failure-prediction.git
cd Build-failure-prediction
python -m pip install -r requirements.txt
```

---

## 4. Repository structure

```
Build-Failure Prediction/
├── src/build_failure/
│   ├── config.py             # paths, feature contract, reproducibility constants
│   ├── data_generator.py     # synthetic CI build dataset
│   ├── preprocessing.py      # cleaning + stratified splitting
│   ├── train.py              # candidate models, CV selection, persistence
│   ├── evaluate.py           # metrics + JSON report
│   ├── predict.py            # single-record & batch inference
│   └── api.py                # Flask REST API
├── tests/                    # one test module per feature
├── .github/workflows/ci.yml  # CI/CD pipeline
├── Dockerfile / .dockerignore
├── Makefile
├── main.py                   # CLI entry point
└── docs/LEARNING.md          # this document
```

---

## 5. Step-by-step walkthrough

### Step 1 — Project scaffolding
**Commit:** `bb27e23` — *chore: scaffold project structure, config and test harness*

**Files:** `.gitignore`, `README.md`, `requirements.txt`, `pyproject.toml`,
`src/build_failure/__init__.py`, `src/build_failure/config.py`, `tests/test_package.py`

**What was done**
- Initialised the Git repository on the `main` branch and pointed it at the
  GitHub remote.
- Chose a **`src/` layout** so the package must be explicitly importable —
  this prevents tests from accidentally importing the local directory instead of
  the installed package.
- Created `config.py` as the **single source of truth** for output paths, the
  random seed, and the `FEATURE_COLUMNS` contract. Every later module imports
  from here, so the data generator, trainer and API can never disagree about the
  schema.
- Configured pytest in `pyproject.toml` using the `pythonpath` option so `src/`
  and the project root are importable without an editable install.

**Verification**

```bash
python -m pytest     # 3 passed
```

**Learning:** Defining a feature contract *before* writing any model code
removes an entire class of "column mismatch" bugs later. Keeping seeds and paths
in one module is what makes the whole pipeline reproducible.

---

### Step 2 — Synthetic build dataset
**Commit:** `ada9571` — *feat: add synthetic CI build dataset generator with reproducible seeding*

**Files:** `src/build_failure/data_generator.py`, `tests/test_data_generator.py`

**What was done**
- Real CI logs are not shareable, so `generate_build_dataset()` fabricates a
  statistically plausible dataset of build records.
- Feature distributions were chosen to look realistic: right-skewed change sizes
  (`gamma`), bounded coverage (`normal` clipped to `[0.03, 0.99]`), and counts
  from `Poisson` / `binomial` distributions.
- Crucially, the label is **not random**. It is sampled from a logistic function
  of standardised features plus label noise:

  ```python
  logit = (-0.35
           + 0.95 * z(lines_added)
           + 0.75 * z(files_changed)
           - 1.15 * z(test_coverage)     # coverage protects
           + 0.90 * z(prior_failures)
           + 0.70 * dependencies_changed
           + 0.60 * z(lint_warnings)
           - 0.30 * z(changed_test_files)
           ...)
  build_failed = rng.random(n) < sigmoid(logit)
  ```

  This encodes intuitive DevOps relationships, so a model **can genuinely learn
  signal** rather than memorise noise — an important property for a demo dataset.
- All randomness flows from a single `numpy.random.default_rng(random_state)`.

**Verification**

```bash
python -m pytest tests/test_data_generator.py   # 9 passed
python -c "from build_failure.data_generator import generate_build_dataset as g; \
           print(g(2000)['build_failed'].mean())"   # ~0.48
```

**Learning:** A generated dataset is only useful if the target is *derived from*
the features. Two identical seeds must also produce byte-identical tables —
that is what makes experiment results trustworthy.

---

### Step 3 — Cleaning and splitting
**Commit:** `4236159` — *feat: add data cleaning and stratified train/test splitting*

**Files:** `src/build_failure/preprocessing.py`, `tests/test_preprocessing.py`

**What was done**
- `clean_data()` drops duplicates and nulls, rejects rows whose label is not
  `0/1`, clips physically impossible values (`test_coverage` to `[0,1]`,
  `commit_hour` to `[0,23]`), and coerces every feature to numeric.
- It also **selects only the contracted columns**, which prevents leakage (e.g.
  an ID column slipping into training).
- `train_test_split_data()` uses `stratify=target` so the failure/success ratio is
  preserved in both splits — essential for an imbalanced-ish classification
  problem.
- `prepare_datasets()` provides a one-call load→clean→split path for the CLI.

**Verification**

```bash
python -m pytest tests/test_preprocessing.py   # 9 passed
```

**Learning:** Data cleaning is a *correctness* feature, not boilerplate. Clipping
and schema enforcement are cheap insurance against silent model degradation.
Stratification also stops a small test set from accidentally containing no
failures at all.

---

### Step 4 — Training pipeline
**Commit:** `4d33f47` — *feat: add cross-validated model training pipeline with persistence*

**Files:** `src/build_failure/train.py`, `tests/test_train.py`

**What was done**
- Defined three candidate classifiers: **Logistic Regression** (interpretable
  baseline), **Random Forest** and **Gradient Boosting** (non-linear ensembles).
- Every candidate is wrapped in the *same* `Pipeline(StandardScaler →
  classifier)`, keeping the comparison fair (scaling matters for the linear
  model, and is harmless for trees).
- `cross_validate_models()` ranks candidates by **mean ROC-AUC across stratified
  K-fold CV** — a more honest estimate than a single split.
- `train_best_model()` refits the winner on all training data and returns the
  pipeline, the winning name, and the full score table.
- `save_model()` / `load_model()` persist the *entire pipeline* (scaler +
  classifier) with joblib, so the serving code can never forget to preprocess.

**Verification**

```bash
python -m pytest tests/test_train.py   # 7 passed
```

**Learning:** Cross-validation plus a shared preprocessing pipeline is the
minimum bar for a defensible model-selection step. Saving the whole pipeline
(not just the estimator) is the single most common MLOps mistake avoided here.

---

### Step 5 — Evaluation and reporting
**Commit:** `2e542d4` — *feat: add model evaluation metrics and JSON report generation*

**Files:** `src/build_failure/evaluate.py`, `tests/test_evaluate.py`

**What was done**
- `evaluate_model()` computes accuracy, precision, recall, F1 and ROC-AUC, plus
  an explicit confusion matrix broken out into TN/FP/FN/TP.
- Used `zero_division=0` so metrics stay well-defined on tiny test sets instead
  of returning `NaN`.
- `save_metrics()` writes a machine-readable `reports/metrics.json` — the
  artefact the CI job uploads — and `format_report()` prints a human-readable
  block for the terminal.

**Actual result on the held-out 20% test set (5 000 generated builds):**

```
accuracy   : 0.7920        precision : 0.7726
recall     : 0.8167        f1        : 0.7941
roc-auc    : 0.8776
TN=391  FP=118  FN=90  TP=401
```

**Learning:** Accuracy alone is misleading for failure prediction. Recall (how
many real failures we catch) and ROC-AUC (ranking quality) matter more, because
the cost of a missed failure is higher than the cost of an extra warning.

---

### Step 6 — Prediction API and CLI
**Commit:** `ac126af` — *feat: add prediction API and CLI (generate-data, train, evaluate, predict)*

**Files:** `src/build_failure/predict.py`, `main.py`, `tests/test_predict.py`, `tests/test_cli.py`, `pyproject.toml`

**What was done**
- `predict_build()` scores a single record and returns a decision, a
  probability, and a **risk bucket** (`low` < 0.35, `medium` < 0.65, else `high`).
  `validate_record()` fails fast with a clear message when a feature is missing.
- `predict_batch()` scores a dataframe and appends
  `failure_probability`, `predicted_failure` and `risk_level` columns.
- `main.py` exposes the whole pipeline as a four-command CLI:

  ```bash
  python main.py generate-data --samples 5000
  python main.py train --cv 5
  python main.py evaluate
  python main.py predict --json '{"lines_added": 900, ...}'
  ```

- The CLI is written as a testable function (`main(argv)`) rather than a bare
  script, so the tests can drive it end-to-end.

**Verification**

```bash
python -m pytest tests/test_predict.py tests/test_cli.py   # 13 passed
```

Example output for a deliberately risky build:

```json
{ "build_failed": true, "failure_probability": 1.0, "risk_level": "high" }
```

**Learning:** Turning a model into a *decision* (a risk bucket a pipeline can gate
on) is what makes it useful in DevOps. Also: designing CLIs as importable
functions makes them far easier to test than subprocess calls.

---

### Step 7 — Flask REST API
**Commit:** `4353500` — *feat: add Flask REST API for single and batch build-risk scoring*

**Files:** `src/build_failure/api.py`, `tests/test_api.py`

**What was done**
- Used the **application-factory pattern** (`create_app(model_path)`) so tests can
  inject a model trained in a temporary directory, instead of relying on a
  globally-loaded model that may not exist.
- Endpoints:

  | Method | Path | Purpose |
  | --- | --- | --- |
  | GET | `/health` | liveness probe + model path |
  | POST | `/predict` | score one build record |
  | POST | `/predict/batch` | score a list of records |

- Proper HTTP semantics: `400` for a malformed body, `422` for a well-formed
  body missing required features, `200` on success.

```bash
python -m build_failure.api    # http://127.0.0.1:5000
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"lines_added": 320, "test_coverage": 0.42, ...}'
```

**Verification**

```bash
python -m pytest tests/test_api.py   # 6 passed
```

**Learning:** The factory pattern plus Flask's `test_client()` gives fast,
model-independent API tests with no network or server process involved.

---

### Step 8 — CI/CD pipeline
**Commit:** `ff1f74b` — *ci: add GitHub Actions workflow for tests and training smoke check*

**Files:** `.github/workflows/ci.yml`

**What was done**
- A two-job workflow triggered on pushes and pull requests to `main`:
  1. **`test`** — a matrix job running the full test suite on **Python 3.10, 3.11
     and 3.12** with pip caching, so version-specific breakage is caught early.
  2. **`train-smoke`** — depends on `test`; generates 5 000 records, trains with
     5-fold CV, evaluates, and **uploads `reports/metrics.json` as a build
     artefact** via `actions/upload-artifact`.
- `fail-fast: false` so one failing Python version does not hide the others.

**Verification:** the workflow YAML was parsed locally with `yaml.safe_load` and
the expected jobs/steps were asserted by `tests/test_infrastructure.py`.

**Learning:** A CI pipeline should test *and* prove the pipeline still trains.
Archiving the metrics JSON creates a lightweight model-quality audit trail on
every run — a small but real MLOps practice.

---

### Step 9 — Containerisation and task runner
**Commit:** `e809ffd` — *build: add Docker image with baked model, healthcheck and Makefile shortcuts*

**Files:** `Dockerfile`, `.dockerignore`, `Makefile`, `tests/test_infrastructure.py`

**What was done**
- A `python:3.11-slim` image that installs dependencies **before** copying the
  source, so the expensive dependency layer is cached across code changes.
- The image is **self-contained**: the build generates data, trains with 5-fold
  CV and evaluates, baking a deterministic model into the image.
- Added a `HEALTHCHECK` that calls `/health` using `urllib` (the slim base image
  has no `curl`).
- `.dockerignore` keeps `data/`, `models/`, `reports/` and `.git/` out of the
  build context.
- A `Makefile` wraps the common tasks: `make install | data | train | evaluate |
  test | api | docker-build | docker-run | clean`.

**Verification**

```bash
python -m pytest tests/test_infrastructure.py   # 4 passed
```

**Learning:** Container conformance tests — asserting the `EXPOSE`, `HEALTHCHECK`
and `FROM` contracts, and that Makefile recipes are tab-indented — catch broken
infrastructure in the same test run as the application code.

---

### Step 10 — Documentation
**Commit:** *(this file)* — *docs: add assignment learning document*

The final commit adds this write-up so the repository is self-documenting.

---

## 6. Results

Classification performance on the held-out test set (1 000 builds):

| Metric | Value |
| --- | --- |
| Accuracy | 0.7920 |
| Precision | 0.7726 |
| Recall | 0.8167 |
| F1 score | 0.7941 |
| ROC-AUC | 0.8776 |

Model selection (mean 5-fold CV ROC-AUC):

| Candidate | CV ROC-AUC |
| --- | --- |
| Logistic Regression | 0.8792 ← selected |
| Gradient Boosting | 0.8546 |
| Random Forest | 0.8535 |

Logistic Regression winning is expected and instructive: the synthetic label was
generated from a **linear** logit, so the linear model matches the true data
generating process and generalises best. On real, noisier CI data the ensembles
would likely overtake it.

---

## 7. Reproducing everything from scratch

```bash
python -m pip install -r requirements.txt

python main.py generate-data --samples 5000   # deterministic synthetic data
python main.py train --cv 5                    # CV-select and save best model
python main.py evaluate                        # metrics -> reports/metrics.json
python main.py predict --json '{"lines_added": 900, "test_coverage": 0.10,
    "files_changed": 20, "lines_deleted": 400, "prior_failures": 4,
    "dependencies_changed": 1, "commit_hour": 23, "is_weekend": 1,
    "distinct_authors": 5, "changed_test_files": 0, "avg_review_time_min": 3,
    "open_issues": 40, "test_count": 30, "lint_warnings": 25}'

python -m pytest                               # 57 passed
```

Or, with the shortcuts: `make data && make train && make evaluate && make test`.

---

## 8. Key learnings

1. **Define the data contract first.** A single `config.py` holding the feature
   list and seeds kept six modules consistent and made the pipeline reproducible.
2. **Generated data must encode real signal.** Deriving labels from a noisy
   logistic function of the features produced a dataset a model can actually
   learn from — and made the evaluation numbers meaningful.
3. **Stratify and clean before you split.** Duplicate removal, range clipping
   and stratified splitting directly affect the validity of every metric.
4. **Compare models fairly.** One shared `Pipeline` for every candidate plus
   cross-validated ROC-AUC is the minimum bar for honest model selection.
5. **Persist the whole pipeline,** not just the estimator — otherwise serving
   silently skips preprocessing.
6. **Accuracy is not enough.** Recall and ROC-AUC reflect what actually matters
   when missing a failure is expensive.
7. **Make models decision-ready.** A probability plus a `low/medium/high` risk
   bucket is what a CI job can act on.
8. **Design for testability.** CLI-as-function and the Flask factory pattern made
   both the command line and the API easy to test without side effects.
9. **CI should prove the pipeline trains,** not just that unit tests pass —
   and archiving metrics creates a quality audit trail.
10. **Container and Make targets are code too;** they deserve conformance tests.
11. **One feature per commit, gated on green tests,** produced a history that is
    readable, reviewable and revertible — and directly reusable as this document.

---

## 9. Challenges encountered

| Challenge | Resolution |
| --- | --- |
| No shareable real CI dataset | Generated a synthetic dataset whose labels are derived from the features, so signal is learnable |
| `python main.py` could not import the `src/` package | Added a path bootstrap in `main.py` **and** pytest `pythonpath` config |
| Making the API testable without a pre-existing model file | Adopted the Flask application-factory pattern and injected a temp-trained model |
| Slim Docker base has no `curl` | Wrote the `HEALTHCHECK` using `python -c` and `urllib.request` instead |
| "Which model is best?" | Ranked all candidates with stratified 5-fold CV ROC-AUC in a single command |

---

## 10. Possible future improvements

- Replace the synthetic generator with a real public build dataset (e.g. Travis
  CI build logs) and re-train.
- Add `SHAP` explanations so authors see *why* their commit is high-risk.
- Persist metrics history and track drift between training runs.
- Deploy the Flask image to a registry and add a staging deploy job to CI.
- Add `pre-commit` hooks (ruff, black) and enforce a coverage threshold.

---

## 11. Submission links

- **GitHub repository:** <https://github.com/RejuthaSree/Build-failure-prediction>
- **Pipeline code:** `src/build_failure/`
- **Tests (57 passing):** `tests/`
- **CI configuration:** `.github/workflows/ci.yml`
- **Container definition:** `Dockerfile`
