"""Command-line interface for the Build-Failure Prediction pipeline.

Usage
-----
    python main.py generate-data --samples 5000
    python main.py train --cv 5
    python main.py evaluate
    python main.py predict --json '{"lines_added": 320, "test_coverage": 0.42}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the src/ package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from build_failure import config  # noqa: E402
from build_failure.data_generator import generate_build_dataset, save_dataset  # noqa: E402
from build_failure.evaluate import evaluate_model, format_report, save_metrics  # noqa: E402
from build_failure.predict import load_default_model, predict_batch, predict_build  # noqa: E402
from build_failure.preprocessing import (  # noqa: E402
    clean_data,
    load_dataset,
    split_features_target,
    train_test_split_data,
)
from build_failure.train import save_model, train_best_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-failure",
        description="Predict whether a CI/CD build will fail.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="create the synthetic dataset")
    gen.add_argument("--samples", type=int, default=config.DEFAULT_N_SAMPLES)
    gen.add_argument("--output", type=Path, default=config.RAW_DATA_PATH)
    gen.add_argument("--seed", type=int, default=config.RANDOM_STATE)

    train = sub.add_parser("train", help="train and persist the best model")
    train.add_argument("--data", type=Path, default=config.RAW_DATA_PATH)
    train.add_argument("--cv", type=int, default=5)
    train.add_argument("--output", type=Path, default=config.MODEL_PATH)

    ev = sub.add_parser("evaluate", help="score the model on the held-out split")
    ev.add_argument("--data", type=Path, default=config.RAW_DATA_PATH)
    ev.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ev.add_argument("--metrics", type=Path, default=config.METRICS_PATH)

    pred = sub.add_parser("predict", help="score one build or a CSV of builds")
    pred.add_argument("--model", type=Path, default=config.MODEL_PATH)
    pred.add_argument("--json", type=str, default=None, help="a single build as JSON")
    pred.add_argument("--input", type=Path, default=None, help="CSV of builds")
    pred.add_argument("--output", type=Path, default=None, help="where to write CSV results")

    return parser


def cmd_generate_data(args: argparse.Namespace) -> int:
    config.ensure_directories()
    frame = generate_build_dataset(n_samples=args.samples, random_state=args.seed)
    path = save_dataset(frame, args.output)
    print(f"Generated {len(frame)} build records -> {path}")
    print(f"Observed failure rate: {frame[config.TARGET_COLUMN].mean():.3f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    frame = clean_data(load_dataset(args.data))
    features, target = split_features_target(frame)
    x_train, _, y_train, _ = train_test_split_data(features, target)

    pipeline, best_name, scores = train_best_model(x_train, y_train, cv=args.cv)
    path = save_model(pipeline, args.output)

    print("Cross-validated ROC-AUC by candidate model:")
    for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        marker = "  <- selected" if name == best_name else ""
        print(f"  {name:<20} {score:.4f}{marker}")
    print(f"Saved best model ({best_name}) -> {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    frame = clean_data(load_dataset(args.data))
    features, target = split_features_target(frame)
    _, x_test, _, y_test = train_test_split_data(features, target)

    model = load_default_model(args.model)
    metrics = evaluate_model(model, x_test, y_test)
    print(format_report(metrics))
    path = save_metrics(metrics, args.metrics)
    print(f"\nMetrics written -> {path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_default_model(args.model)

    if args.json:
        record = json.loads(args.json)
        print(json.dumps(predict_build(model, record), indent=2))
        return 0

    if args.input:
        frame = load_dataset(args.input)
        scored = predict_batch(model, frame)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output, index=False)
            print(f"Scored {len(scored)} builds -> {args.output}")
        else:
            print(scored.to_string(index=False))
        return 0

    print("Provide either --json '<record>' or --input <csv>.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
