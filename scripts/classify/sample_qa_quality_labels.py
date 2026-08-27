#!/usr/bin/env python3
"""Create a QA/social labeling sample for the qa_quality classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.io import ensure_parent  # noqa: E402


METADATA_COLUMNS = (
    "source_id",
    "record_id",
    "content_hash",
    "title",
    "content_length",
    "score",
    "answer_count",
    "has_accepted_answer",
    "closed",
    "tags",
    "source_url",
    "license",
)
FULL_CONTENT_COLUMNS = (*METADATA_COLUMNS, "content")
QA_FAMILY_LABELS = {
    "stackoverflow": "stackoverflow",
    "stackexchange": "stackexchange",
    "reddit": "reddit",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    datasets = _load_family_datasets(args.input)
    metadata = {
        family: _metadata_frame(family, dataset)
        for family, dataset in datasets.items()
    }

    used_keys: set[tuple[str, str, str]] = set()
    selected_frames = [
        _sample_random(
            metadata["stackoverflow"],
            args.stackoverflow_size,
            rng=rng,
            used_keys=used_keys,
            sample_bucket="stackoverflow_random",
            edge_case_reason="",
        ),
        _sample_balanced_by_source(
            metadata["stackexchange"],
            args.stackexchange_size,
            rng=rng,
            used_keys=used_keys,
            sample_bucket="stackexchange_balanced_random",
        ),
        _sample_balanced_by_source(
            metadata["reddit"],
            args.reddit_size,
            rng=rng,
            used_keys=used_keys,
            sample_bucket="reddit_balanced_random",
        ),
    ]

    combined_metadata = pd.concat(metadata.values(), ignore_index=True)
    selected_frames.append(
        _sample_edge_cases(
            combined_metadata,
            args.edge_size,
            rng=rng,
            used_keys=used_keys,
        )
    )
    selected_metadata = pd.concat(
        [frame for frame in selected_frames if not frame.empty],
        ignore_index=True,
    )
    if selected_metadata.empty:
        raise ValueError("No rows were selected for labeling")

    full_rows = _materialize_content(selected_metadata, datasets)
    full_rows = _finalize_output(full_rows, csv_content_chars=args.csv_content_chars)

    ensure_parent(args.output_parquet)
    full_rows.to_parquet(args.output_parquet, index=False)
    if args.output_csv:
        ensure_parent(args.output_csv)
        csv_rows = full_rows.copy()
        csv_rows["content"] = csv_rows["content_for_labeling"]
        csv_rows.to_csv(args.output_csv, index=False)

    summary = (
        full_rows.groupby(["sample_bucket", "source_family", "source_id"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["sample_bucket", "source_family", "source_id"])
    )
    print(f"Wrote {len(full_rows):,} labeling rows to {args.output_parquet}")
    if args.output_csv:
        print(f"Wrote labeling CSV to {args.output_csv}")
    print(summary.to_string(index=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/training-clean-v2/normalized"),
        help="Hive-partitioned normalized corpus directory.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/classifier-labels/qa_quality_labeling_sample.parquet"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/classifier-labels/qa_quality_labeling_sample.csv"),
        help="Optional CSV copy for manual labeling.",
    )
    parser.add_argument("--stackoverflow-size", type=int, default=500)
    parser.add_argument("--stackexchange-size", type=int, default=500)
    parser.add_argument("--reddit-size", type=int, default=500)
    parser.add_argument("--edge-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--csv-content-chars",
        type=int,
        default=20_000,
        help="Truncate CSV content cells to this many chars. Parquet keeps full content.",
    )
    return parser


def _load_family_datasets(root: Path) -> dict[str, ds.Dataset]:
    source_dirs = [path for path in sorted(root.glob("source_id=*")) if path.is_dir()]
    if not source_dirs:
        raise FileNotFoundError(f"No source_id partitions found under {root}")

    families = {
        "stackoverflow": [
            path for path in source_dirs if path.name == "source_id=stackoverflow"
        ],
        "stackexchange": [
            path for path in source_dirs if path.name.startswith("source_id=stackexchange-")
        ],
        "reddit": [
            path for path in source_dirs if path.name.startswith("source_id=reddit-")
        ],
    }
    missing = [family for family, dirs in families.items() if not dirs]
    if missing:
        raise FileNotFoundError(
            "Missing QA source partitions: " + ", ".join(sorted(missing))
        )
    return {
        family: ds.dataset(
            [str(file) for directory in dirs for file in sorted(directory.glob("**/*.parquet"))],
            format="parquet",
        )
        for family, dirs in families.items()
    }


def _metadata_frame(family: str, dataset: ds.Dataset) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in dataset.schema.names]
    missing = [column for column in ("source_id", "record_id", "content_hash") if column not in columns]
    if missing:
        raise ValueError(f"{family} dataset is missing key columns: {', '.join(missing)}")
    frame = dataset.to_table(columns=columns).to_pandas()
    frame["source_family"] = family
    frame["_family_row_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _sample_random(
    frame: pd.DataFrame,
    n: int,
    *,
    rng: np.random.Generator,
    used_keys: set[tuple[str, str, str]],
    sample_bucket: str,
    edge_case_reason: str,
) -> pd.DataFrame:
    candidates = _without_used_keys(frame, used_keys)
    if candidates.empty or n <= 0:
        return candidates.head(0).copy()
    sampled = candidates.sample(
        n=min(n, len(candidates)),
        replace=False,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).copy()
    sampled["sample_bucket"] = sample_bucket
    sampled["edge_case_reason"] = edge_case_reason
    _mark_used(sampled, used_keys)
    return sampled


def _sample_balanced_by_source(
    frame: pd.DataFrame,
    n: int,
    *,
    rng: np.random.Generator,
    used_keys: set[tuple[str, str, str]],
    sample_bucket: str,
) -> pd.DataFrame:
    candidates = _without_used_keys(frame, used_keys)
    if candidates.empty or n <= 0:
        return candidates.head(0).copy()
    groups = {
        source_id: group
        for source_id, group in candidates.groupby("source_id", sort=True)
    }
    quotas = _balanced_quotas({key: len(value) for key, value in groups.items()}, n)
    sampled_parts = []
    for source_id, quota in quotas.items():
        if quota <= 0:
            continue
        group = groups[source_id]
        sampled_parts.append(
            group.sample(
                n=min(quota, len(group)),
                replace=False,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
        )
    if not sampled_parts:
        return candidates.head(0).copy()
    sampled = pd.concat(sampled_parts, ignore_index=True).copy()
    sampled["sample_bucket"] = sample_bucket
    sampled["edge_case_reason"] = ""
    _mark_used(sampled, used_keys)
    return sampled


def _sample_edge_cases(
    frame: pd.DataFrame,
    n: int,
    *,
    rng: np.random.Generator,
    used_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    if n <= 0:
        return frame.head(0).copy()
    buckets = [
        ("edge_shortest_content", lambda data: data.nsmallest(max(200, n), "content_length")),
        ("edge_longest_content", lambda data: data.nlargest(max(200, n), "content_length")),
        ("edge_lowest_score", lambda data: data.nsmallest(max(200, n), "score")),
        ("edge_highest_score", lambda data: data.nlargest(max(200, n), "score")),
        ("edge_no_answers", lambda data: data[data["answer_count"].fillna(0) == 0]),
        ("edge_closed", lambda data: data[data["closed"].fillna(False).astype(bool)]),
    ]
    per_bucket = max(1, n // len(buckets))
    sampled_parts = []
    for reason, selector in buckets:
        candidates = _without_used_keys(frame, used_keys)
        if candidates.empty:
            break
        candidates = _numeric_metadata(candidates)
        pool = selector(candidates)
        sampled = _sample_random(
            pool,
            per_bucket,
            rng=rng,
            used_keys=used_keys,
            sample_bucket="mixed_edge_cases",
            edge_case_reason=reason,
        )
        if not sampled.empty:
            sampled_parts.append(sampled)

    selected = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else frame.head(0).copy()
    remainder = n - len(selected)
    if remainder > 0:
        filler = _sample_random(
            frame,
            remainder,
            rng=rng,
            used_keys=used_keys,
            sample_bucket="mixed_edge_cases",
            edge_case_reason="edge_random_filler",
        )
        selected = pd.concat([selected, filler], ignore_index=True)
    return selected.head(n).copy()


def _materialize_content(
    selected: pd.DataFrame,
    datasets: dict[str, ds.Dataset],
) -> pd.DataFrame:
    rows = []
    for family, group in selected.groupby("source_family", sort=False):
        indices = group["_family_row_index"].astype(np.int64).tolist()
        columns = [
            column
            for column in FULL_CONTENT_COLUMNS
            if column in datasets[family].schema.names
        ]
        table = datasets[family].take(indices, columns=columns)
        content_frame = table.to_pandas()
        group = group.reset_index(drop=True)
        for column in ("sample_bucket", "edge_case_reason"):
            content_frame[column] = group[column]
        content_frame["source_family"] = family
        rows.append(content_frame)
    return pd.concat(rows, ignore_index=True)


def _finalize_output(frame: pd.DataFrame, *, csv_content_chars: int) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "qa_quality_label", "")
    output.insert(1, "label_notes", "")
    output["content"] = output["content"].fillna("").astype(str)
    output["content_preview"] = output["content"].map(lambda value: _preview(value, 1_000))
    output["content_for_labeling"] = output["content"].map(
        lambda value: _preview(value, csv_content_chars)
    )
    output["csv_content_truncated"] = output["content"].str.len() > csv_content_chars
    output = output.sort_values(
        ["sample_bucket", "source_family", "source_id", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    output.insert(0, "sample_index", np.arange(1, len(output) + 1))
    preferred = [
        "sample_index",
        "qa_quality_label",
        "label_notes",
        "sample_bucket",
        "edge_case_reason",
        "source_family",
        "source_id",
        "record_id",
        "content_hash",
        "title",
        "content_length",
        "score",
        "answer_count",
        "has_accepted_answer",
        "closed",
        "tags",
        "source_url",
        "license",
        "content_preview",
        "content_for_labeling",
        "csv_content_truncated",
        "content",
    ]
    return output[[column for column in preferred if column in output.columns]]


def _balanced_quotas(group_sizes: dict[str, int], n: int) -> dict[str, int]:
    remaining = min(n, sum(group_sizes.values()))
    quotas = {key: 0 for key in group_sizes}
    active = {key for key, size in group_sizes.items() if size > 0}
    while remaining > 0 and active:
        increment = max(1, remaining // len(active))
        progressed = False
        for key in sorted(active):
            available = group_sizes[key] - quotas[key]
            take = min(increment, available, remaining)
            if take <= 0:
                continue
            quotas[key] += take
            remaining -= take
            progressed = True
            if quotas[key] >= group_sizes[key]:
                active.remove(key)
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _numeric_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in ("content_length", "score", "answer_count"):
        if column in numeric:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _without_used_keys(
    frame: pd.DataFrame,
    used_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    if not used_keys:
        return frame.copy()
    mask = ~frame.apply(lambda row: _key(row) in used_keys, axis=1)
    return frame[mask].copy()


def _mark_used(frame: pd.DataFrame, used_keys: set[tuple[str, str, str]]) -> None:
    used_keys.update(_key(row) for _, row in frame.iterrows())


def _key(row: Any) -> tuple[str, str, str]:
    return (
        str(row["source_id"]),
        str(row["record_id"]),
        str(row["content_hash"]),
    )


def _preview(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n[TRUNCATED]"


if __name__ == "__main__":
    raise SystemExit(main())
