#!/usr/bin/env python3
"""Build the derived Tier 1 multi-hop reasoning dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from ingest.derived.tier1_links import build_tier1_reasoning_dataset


def _path_arg(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build linked Tier 1 CVE -> CWE -> CAPEC -> ATT&CK reasoning data.",
    )
    parser.add_argument(
        "--data-dir",
        type=_path_arg,
        default=Path("data/training-clean-v2/normalized"),
        help="Root normalized data directory containing Tier 1 source partitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2"),
        help="Directory for nodes.parquet, edges.parquet, chains.parquet, manifest.json.",
    )
    parser.add_argument(
        "--max-chains",
        type=int,
        help="Optional cap for quick development builds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_tier1_reasoning_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_chains=args.max_chains,
    )
    print(f"nodes: {result.nodes:,}")
    print(f"edges: {result.edges:,}")
    print(f"chains: {result.chains:,}")
    print(f"chains with Sigma: {result.chains_with_sigma:,}")
    print(f"detection chains: {result.detection_chains:,}")
    print(f"complete KEV chains: {result.complete_kev_chains:,}")
    print(f"complete KEV detection chains: {result.complete_kev_detection_chains:,}")
    print(f"output: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
