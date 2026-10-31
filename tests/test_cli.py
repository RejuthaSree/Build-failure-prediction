"""End-to-end tests for the command-line interface in ``main.py``."""

import json

from build_failure import config
from main import build_parser, main


def test_parser_requires_a_command() -> None:
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit as exc:  # argparse exits with code 2 on error
        assert exc.code == 2
    else:  # pragma: no cover - defensive
        raise AssertionError("expected argparse to reject an empty command")


def test_generate_data_writes_dataset(tmp_path) -> None:
    out = tmp_path / "raw" / "builds.csv"
    assert main(["generate-data", "--samples", "250", "--output", str(out)]) == 0
    assert out.exists()


def test_train_then_evaluate_then_predict(tmp_path) -> None:
    data = tmp_path / "builds.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    assert main(["generate-data", "--samples", "400", "--output", str(data)]) == 0
    assert main(
        ["train", "--data", str(data), "--cv", "3", "--output", str(model_path)]
    ) == 0
    assert model_path.exists()

    assert main(
        [
            "evaluate",
            "--data", str(data),
            "--model", str(model_path),
            "--metrics", str(metrics_path),
        ]
    ) == 0
    assert metrics_path.exists()

    record = {column: 1 for column in config.FEATURE_COLUMNS}
    assert main(
        ["predict", "--model", str(model_path), "--json", json.dumps(record)]
    ) == 0


def test_predict_requires_input(tmp_path) -> None:
    data = tmp_path / "builds.csv"
    model_path = tmp_path / "model.joblib"
    main(["generate-data", "--samples", "200", "--output", str(data)])
    main(["train", "--data", str(data), "--cv", "3", "--output", str(model_path)])
    # No --json and no --input means the command cannot do anything.
    assert main(["predict", "--model", str(model_path)]) == 2


def test_predict_batch_writes_csv(tmp_path) -> None:
    data = tmp_path / "builds.csv"
    model_path = tmp_path / "model.joblib"
    scored_path = tmp_path / "scored.csv"

    main(["generate-data", "--samples", "200", "--output", str(data)])
    main(["train", "--data", str(data), "--cv", "3", "--output", str(model_path)])
    assert main(
        [
            "predict",
            "--model", str(model_path),
            "--input", str(data),
            "--output", str(scored_path),
        ]
    ) == 0
    assert scored_path.exists()
