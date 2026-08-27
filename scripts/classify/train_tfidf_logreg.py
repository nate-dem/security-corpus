#!/usr/bin/env python3
"""Train a TF-IDF plus logistic regression classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.schema import DEFAULT_TASKS, TEXT_COLUMN
from classify.tfidf_logreg import TfidfLogRegConfig, train_model


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    class_weight = None if args.no_class_weight else "balanced"
    config = TfidfLogRegConfig(
        task_name=args.task,
        text_column=args.text_column,
        label_column=args.label_column,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        class_weight=class_weight,
        max_iter=args.max_iter,
        random_state=args.random_state,
        validation_fraction=args.validation_fraction,
    )

    train_model(args.labels, args.model_dir, config)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="CSV or Parquet labels file.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory for model artifacts.")
    parser.add_argument(
        "--task",
        choices=sorted(DEFAULT_TASKS),
        default="security_relevance",
        help="Classifier task to train.",
    )
    parser.add_argument("--text-column", default=TEXT_COLUMN)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=500_000)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Optional stratified validation split used for metrics.json. "
            "Set to 0 to skip validation."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
