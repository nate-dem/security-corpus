#!/usr/bin/env python3
"""
Download CTIBench from HuggingFace and save each task split as a parquet file.

CTIBench (NeurIPS 2024) — 5 tasks:
    cti_mcq   Cyber Threat Intelligence Multiple Choice Questions   (~2,500 samples)
    cti_rcm   Recommended Course of Action Matching                 (~1,000 samples)
    cti_vsp   Vulnerability Severity Prediction                     (~1,000 samples)
    cti_ate   ATT&CK Technique Extraction                           (~397 samples)
    cti_taa   Threat Actor Attribution                              (~50 samples)

Output:
    data/ctibench/cti_mcq.parquet
    data/ctibench/cti_rcm.parquet
    data/ctibench/cti_vsp.parquet
    data/ctibench/cti_ate.parquet
    data/ctibench/cti_taa.parquet

Usage:
    python3 scripts/download_ctibench.py
    python3 scripts/download_ctibench.py --output data/ctibench --dataset-id AI4Sec/cti-bench
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATASET_ID = "AI4Sec/cti-bench"

# Map local names → HuggingFace config names for AI4Sec/cti-bench.
# Each config has a single "test" split.
_TASK_CONFIGS: dict[str, str] = {
    "cti_mcq": "cti-mcq",
    "cti_rcm": "cti-rcm",
    "cti_vsp": "cti-vsp",
    "cti_ate": "cti-ate",
    "cti_taa": "cti-taa",
}


def download_ctibench(dataset_id: str, output_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required. Install with:\n"
            "  pip install -e '.[fineweb]'\n"
            "  # or: pip install datasets"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for local_name, config_name in _TASK_CONFIGS.items():
        out_path = output_dir / f"{local_name}.parquet"

        logger.info(f"Downloading {dataset_id} / {config_name} ...")
        try:
            ds = load_dataset(dataset_id, config_name, split="test", trust_remote_code=False)
        except Exception as exc:
            logger.error(f"  Failed to download {config_name}: {exc}")
            continue

        table = ds.data.table
        pq.write_table(table, str(out_path), compression="snappy")
        rows = len(table)
        total_rows += rows
        logger.info(f"  {config_name:10s} → {out_path.name}  ({rows:,} rows, columns: {table.column_names})")

    logger.info(f"\nDone. {total_rows:,} total rows across {len(_TASK_CONFIGS)} tasks → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CTIBench task splits to parquet.")
    parser.add_argument(
        "--output", default="data/ctibench",
        help="Output directory (default: data/ctibench)",
    )
    parser.add_argument(
        "--dataset-id", default=_DATASET_ID,
        help=f"HuggingFace dataset ID (default: {_DATASET_ID})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir

    download_ctibench(args.dataset_id, output_dir)


if __name__ == "__main__":
    main()
