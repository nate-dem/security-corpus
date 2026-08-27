#!/usr/bin/env python3
"""Derive binary QA quality labels from 4-class QA quality labels."""

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

from classify.io import ensure_parent  # noqa: E402


SOURCE_LABEL_COLUMN = "qa_quality_label"
BINARY_LABEL_COLUMN = "qa_quality_binary_label"
BINARY_LABEL_SOURCE_COLUMN = "qa_quality_binary_label_source"
BINARY_LABEL_SOURCE_VALUE = "derived_from_qa_quality_label"


def derive_binary_labels(input_path: Path, output_path: Path) -> int:
    """Derive qa_quality_binary_label from qa_quality_label and write the result."""
    frame = _read_labels(input_path)
    _validate_source_labels(frame, input_path)

    output_frame = frame.copy()
    output_frame[BINARY_LABEL_COLUMN] = output_frame[SOURCE_LABEL_COLUMN].map(
        lambda value: 0 if int(value) in {0, 1} else 1
    )
    output_frame[BINARY_LABEL_SOURCE_COLUMN] = BINARY_LABEL_SOURCE_VALUE

    _write_labels(output_frame, output_path)
    return int(len(output_frame))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = derive_binary_labels(args.input, args.output)
    print(f"Wrote {rows} binary QA quality labels to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV or Parquet labels with qa_quality_label in 0..3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV or Parquet labels with derived qa_quality_binary_label.",
    )
    return parser


def _read_labels(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported labels format: {path}. Use .csv, .csv.gz, or .parquet")


def _write_labels(frame, path: Path) -> None:
    ensure_parent(path)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}. Use .csv, .csv.gz, or .parquet")


def _validate_source_labels(frame, path: Path) -> None:
    import pandas as pd

    if SOURCE_LABEL_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing required column {SOURCE_LABEL_COLUMN!r}")
    labels = frame[SOURCE_LABEL_COLUMN]
    if labels.isna().any():
        raise ValueError(f"{SOURCE_LABEL_COLUMN} contains null labels")
    if not pd.api.types.is_numeric_dtype(labels):
        raise ValueError(
            f"{SOURCE_LABEL_COLUMN} must contain numeric 4-class labels 0, 1, 2, or 3"
        )
    numeric = pd.to_numeric(labels, errors="raise")
    integral = numeric.map(lambda value: float(value).is_integer())
    if not bool(integral.all()):
        raise ValueError(f"{SOURCE_LABEL_COLUMN} must contain integer labels 0, 1, 2, or 3")
    invalid = sorted(set(int(value) for value in numeric if int(value) not in {0, 1, 2, 3}))
    if invalid:
        raise ValueError(
            f"{SOURCE_LABEL_COLUMN} must contain labels in 0..3; found invalid labels: {invalid}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
