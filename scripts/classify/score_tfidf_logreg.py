#!/usr/bin/env python3
"""Score corpus Parquet with a TF-IDF plus logistic regression classifier."""

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

from classify.schema import DEFAULT_TASKS, TEXT_COLUMN  # noqa: E402
from classify.tfidf_logreg import TfidfLogRegConfig, score_parquet  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_ids = list(args.source_id)
    source_like = list(args.source_like)
    if args.qa_sources:
        source_ids.append("stackoverflow")
        source_like.extend(["stackexchange-*", "reddit-*"])

    config = TfidfLogRegConfig(
        task_name=args.task,
        text_column=args.text_column,
    )

    score_parquet(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        config=config,
        batch_size=args.batch_size,
        source_ids=source_ids,
        source_like=source_like,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet file or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output sidecar Parquet path.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing model artifacts.")
    parser.add_argument(
        "--task",
        choices=sorted(DEFAULT_TASKS),
        default="security_relevance",
        help="Classifier task to score.",
    )
    parser.add_argument("--text-column", default=TEXT_COLUMN)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Exact source_id to score. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--source-like",
        action="append",
        default=[],
        help="Glob-style source_id pattern to score, e.g. 'reddit-*'. Repeatable.",
    )
    parser.add_argument(
        "--qa-sources",
        action="store_true",
        help="Score only QA/social sources: stackoverflow, stackexchange-*, reddit-*.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
