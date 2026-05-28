#!/usr/bin/env python3
"""
FineWeb cybersecurity subset extraction — Datatrove streaming pipeline with DSIR scoring.

Streams FineWeb records one document at a time directly from HuggingFace
using Datatrove's ParquetReader. No full dataset download is required.
Documents are scored for cyber relevance via DSIR importance weights and
written to three confidence tiers in a single pass.

Pipeline:
    ParquetReader (HuggingFace)
        → QualityGateFilter   (token_count >= 40, language_score >= 0.65)
        → DSIRScoringStep     (loads scorer.pkl, annotates dsir_score + confidence)
        → TieredParquetWriter (routes to confidence=high / medium / background)

DSIR scorer must be built before running this pipeline:
    python3 scripts/build_dsir_target.py
    python3 scripts/fit_dsir.py --sample-size 1000000   # on Marlowe with full FineWeb

Output:
    data/fineweb/normalized/confidence=high/
    data/fineweb/normalized/confidence=medium/
    data/fineweb/normalized/confidence=background/

Pilot (streams ~5k docs spread across all 6 V1.4 snapshots, no download needed):
    python3 scripts/ingest_fineweb.py --max-docs 5000

Single-dump local run (~40M docs from one CC snapshot):
    python3 scripts/ingest_fineweb.py --dump CC-MAIN-2025-26

Full V1.4 Marlowe run (all 6 new snapshots, 64 SLURM tasks each):
    python3 scripts/ingest_fineweb.py --slurm --tasks 64

Install:
    pip install -e ".[fineweb]"
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Generator

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# HuggingFace dataset root — individual dumps sit one level deeper
_HF_BASE = "hf://datasets/HuggingFaceFW/fineweb/data"

# FineWeb V1.4.0 (released 2025-11-07) — 6 new CommonCrawl snapshots (Jan–Jun 2025).
# These are the most recent data in the dataset and the default for this pipeline.
_V14_SNAPSHOTS: tuple[str, ...] = (
    "CC-MAIN-2025-05",   # January 2025
    "CC-MAIN-2025-08",   # February 2025
    "CC-MAIN-2025-13",   # March 2025
    "CC-MAIN-2025-18",   # April 2025
    "CC-MAIN-2025-21",   # May 2025
    "CC-MAIN-2025-26",   # June 2025 (most recent)
)

CONFIDENCE_TIERS = ("high", "medium", "background")
FLUSH_EVERY = 10_000


# ---------------------------------------------------------------------------
# Datatrove pipeline steps
# ---------------------------------------------------------------------------

def _require_datatrove():
    try:
        import datatrove  # noqa: F401
    except ImportError:
        raise ImportError(
            "datatrove is required. Install with:\n"
            "  pip install -e '.[fineweb]'\n"
            "  # or: pip install 'datatrove[io]'"
        )


class QualityGateFilter:
    """
    Datatrove-compatible BaseFilter that drops documents failing cheap gates.

    Tracks per-reason drop counts accessible via .stats after the pipeline runs.
    """

    name = "QualityGateFilter"

    def __init__(self, min_tokens: int = 80, min_language_score: float = 0.65):
        from datatrove.pipeline.filters.base_filter import BaseFilter
        from datatrove.data import Document

        self.min_tokens = min_tokens
        self.min_language_score = min_language_score
        self.drop_counts: Counter = Counter()

        # Build the actual Datatrove filter class dynamically so we can
        # capture self (the outer object) inside it without metaclass issues.
        _min_tokens = min_tokens
        _min_language_score = min_language_score
        _stats = self.drop_counts

        class _Inner(BaseFilter):
            name = "QualityGateFilter"

            def filter(self, doc: Document) -> bool | tuple[bool, str]:
                _stats["seen"] += 1
                if (doc.metadata.get("token_count") or 0) < _min_tokens:
                    _stats["dropped_short"] += 1
                    return False, "short_doc"
                if (doc.metadata.get("language_score") or 0.0) < _min_language_score:
                    _stats["dropped_lang"] += 1
                    return False, "low_language_score"
                _stats["passed"] += 1
                return True

        self._inner = _Inner()

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


class DSIRScoringStep:
    """
    Datatrove-compatible PipelineStep that scores each document with DSIR
    importance weights and routes it to a confidence tier.

    Loads a pre-fitted DSIRScorer from scorer_path at construction time.
    Annotates doc.metadata["dsir_score"] and doc.metadata["confidence"].

    Tier routing:
        dsir_score >= high_threshold   → "high"
        dsir_score >= medium_threshold → "medium"
        otherwise                      → "background"

    Build the scorer first:
        python3 scripts/build_dsir_target.py
        python3 scripts/fit_dsir.py --sample-size 1000000
    """

    name = "DSIRScoringStep"

    def __init__(
        self,
        scorer_path: str,
        high_threshold: float = 0.0,
        medium_threshold: float = -0.3,
    ):
        from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
        from datatrove.data import Document
        from ingest.connectors.fineweb.dsir import DSIRScorer

        scorer = DSIRScorer.load(scorer_path)
        logger.info(f"Loaded DSIRScorer from {scorer_path} ({scorer})")

        _scorer = scorer
        _high = high_threshold
        _medium = medium_threshold

        class _Inner(PipelineStep):
            name = "DSIRScoringStep"

            def run(
                self,
                data: DocumentsPipeline,
                rank: int = 0,
                world_size: int = 1,
            ) -> Generator[Document, None, None]:
                for doc in data:
                    score = _scorer.score(doc.text or "")
                    doc.metadata["dsir_score"] = score
                    if score >= _high:
                        doc.metadata["confidence"] = "high"
                    elif score >= _medium:
                        doc.metadata["confidence"] = "medium"
                    else:
                        doc.metadata["confidence"] = "background"
                    yield doc

        self._inner = _Inner()

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


class TieredParquetWriter:
    """
    Datatrove-compatible PipelineStep that routes documents into
    confidence-tier-specific parquet files using pyarrow.

    One file per tier per task:
        <output_dir>/confidence=<tier>/<filename>

    For multi-task SLURM runs, pass filename="shard_${rank}.parquet"
    and the rank is substituted automatically.
    """

    name = "TieredParquetWriter"

    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "raw.parquet",
        tiers: tuple[str, ...] = CONFIDENCE_TIERS,
    ):
        from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
        from datatrove.data import Document

        _output_dir = Path(output_dir)
        _filename = filename
        _tiers = tiers
        _flush_every = FLUSH_EVERY

        class _Inner(PipelineStep):
            name = "TieredParquetWriter"

            def run(
                self,
                data: DocumentsPipeline,
                rank: int = 0,
                world_size: int = 1,
            ) -> Generator[Document, None, None]:
                fname = _filename.replace("${rank}", f"{rank:04d}")
                buffers: dict[str, list[dict]] = {t: [] for t in _tiers}
                writers: dict[str, pq.ParquetWriter] = {}
                schemas: dict[str, pa.Schema] = {}

                def flush(tier: str) -> None:
                    rows = buffers[tier]
                    if not rows:
                        return
                    table = pa.Table.from_pylist(rows)
                    if tier not in writers:
                        out_path = _output_dir / f"confidence={tier}" / fname
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        schemas[tier] = table.schema
                        writers[tier] = pq.ParquetWriter(
                            str(out_path), table.schema, compression="snappy"
                        )
                    writers[tier].write_table(table)
                    buffers[tier].clear()

                try:
                    for doc in data:
                        tier = doc.metadata.get("confidence", "background")
                        if tier not in _tiers:
                            yield doc
                            continue
                        row = _doc_to_row(doc)
                        buffers[tier].append(row)
                        if len(buffers[tier]) >= _flush_every:
                            flush(tier)
                        yield doc
                finally:
                    for tier in _tiers:
                        flush(tier)
                    for w in writers.values():
                        w.close()

        self._inner = _Inner()

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


# ---------------------------------------------------------------------------
# Row serialization
# ---------------------------------------------------------------------------

def _doc_to_row(doc) -> dict:
    """Flatten a Datatrove Document into a parquet-friendly dict."""
    from datetime import datetime, timezone

    meta = doc.metadata or {}
    return {
        "source_id": "fineweb",
        "source_record_id": doc.id or "",
        "record_id": f"fineweb:{doc.id or ''}",
        "content": doc.text or "",
        "source_url": meta.get("url"),
        "published_at": meta.get("date"),
        "license": "https://commoncrawl.org/terms-of-use",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "dump": meta.get("dump"),
        "language_score": meta.get("language_score"),
        "token_count": meta.get("token_count"),
        "dsir_score": meta.get("dsir_score"),
        "confidence": meta.get("confidence"),
    }


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    hf_path: str,
    output_dir: Path,
    max_docs: int,
    min_tokens: int,
    min_language_score: float,
    scorer_path: str,
    high_threshold: float,
    medium_threshold: float,
    slurm: bool,
    filename: str = "raw.parquet",
) -> list:
    from datatrove.pipeline.readers import ParquetReader

    limit = max_docs if max_docs > 0 else -1

    return [
        ParquetReader(
            hf_path,
            limit=limit,
            text_key="text",
            id_key="id",
        ),
        QualityGateFilter(
            min_tokens=min_tokens,
            min_language_score=min_language_score,
        ),
        DSIRScoringStep(
            scorer_path=scorer_path,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        ),
        TieredParquetWriter(
            output_dir=output_dir,
            filename="shard_${rank}.parquet" if slurm else filename,
        ),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump", default="v1.4",
        help=(
            "'v1.4' (default) — streams all 6 FineWeb V1.4 snapshots "
            "(CC-MAIN-2025-05 through CC-MAIN-2025-26). "
            "'all' — reads every dump in FineWeb. "
            "Or pass a single dump name, e.g. CC-MAIN-2025-26."
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/fineweb/normalized",
        help="Base output directory (default: data/fineweb/normalized)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Stop after this many input documents (0 = no limit). "
             "Use 5000 for a pilot run.",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=40,
        help="Minimum token count to pass quality gate (default: 40).",
    )
    parser.add_argument(
        "--min-language-score", type=float, default=0.65,
    )
    parser.add_argument(
        "--scorer", default="data/dsir/scorer.pkl",
        help=(
            "Path to a fitted DSIRScorer pickle (default: data/dsir/scorer.pkl). "
            "Build with: python3 scripts/build_dsir_target.py && python3 scripts/fit_dsir.py"
        ),
    )
    parser.add_argument(
        "--high-threshold", type=float, default=0.0,
        help="DSIR score >= this → confidence=high (default: 0.0).",
    )
    parser.add_argument(
        "--medium-threshold", type=float, default=-0.3,
        help="DSIR score >= this → confidence=medium (default: -0.3).",
    )
    # SLURM / Marlowe options
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit to SLURM via SlurmPipelineExecutor instead of running locally.",
    )
    parser.add_argument(
        "--tasks", type=int, default=64,
        help="Number of SLURM tasks (default: 64). Only used with --slurm.",
    )
    parser.add_argument(
        "--partition", default="gpu",
        help="SLURM partition (default: gpu). Only used with --slurm.",
    )
    parser.add_argument(
        "--logs-dir", default="logs/fineweb",
        help="Directory for Datatrove logs and stats.",
    )
    args = parser.parse_args()

    _require_datatrove()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)

    # Resolve which dumps to process and the per-dump doc limit.
    if args.dump == "v1.4":
        dumps = list(_V14_SNAPSHOTS)
    elif args.dump == "all":
        dumps = ["all"]   # single pass over the full HF root
    else:
        dumps = [args.dump]

    # Spread --max-docs evenly across dumps so each gets a fair sample.
    # 0 means no limit per dump.
    limit_per_dump = (args.max_docs // len(dumps)) if args.max_docs > 0 else 0

    logger.info(f"Dumps   : {dumps}")
    logger.info(f"Output  : {output_dir}")
    logger.info(
        f"Max docs: {args.max_docs if args.max_docs > 0 else 'unlimited'} "
        f"({limit_per_dump} per dump)" if len(dumps) > 1 else
        f"Max docs: {args.max_docs if args.max_docs > 0 else 'unlimited'}"
    )

    for dump in dumps:
        hf_path = _HF_BASE if dump == "all" else f"{_HF_BASE}/{dump}"
        # Each dump writes to its own parquet file so runs don't overwrite each other.
        # BUCKET_SOURCES in 01_prepare_buckets.py points to the parent directory
        # and globs for all *.parquet files within it.
        filename = f"raw_{dump}.parquet" if len(dumps) > 1 else "raw.parquet"

        logger.info(f"\n--- {dump} ---")
        logger.info(f"  Source : {hf_path}")
        logger.info(f"  File   : {filename}")

        scorer_path = str(repo_root / args.scorer) if not Path(args.scorer).is_absolute() else args.scorer

        pipeline = build_pipeline(
            hf_path=hf_path,
            output_dir=output_dir,
            max_docs=limit_per_dump,
            min_tokens=args.min_tokens,
            min_language_score=args.min_language_score,
            scorer_path=scorer_path,
            high_threshold=args.high_threshold,
            medium_threshold=args.medium_threshold,
            slurm=args.slurm,
            filename=filename,
        )

        if args.slurm:
            from datatrove.executor import SlurmPipelineExecutor
            executor = SlurmPipelineExecutor(
                pipeline=pipeline,
                tasks=args.tasks,
                time="12:00:00",
                partition=args.partition,
                logging_dir=f"{args.logs_dir}/{dump}",
                job_name=f"fineweb_{dump}",
            )
            logger.info(f"  Submitting {args.tasks} SLURM tasks to '{args.partition}'")
        else:
            from datatrove.executor import LocalPipelineExecutor
            executor = LocalPipelineExecutor(
                pipeline=pipeline,
                tasks=1,
                logging_dir=f"{args.logs_dir}/{dump}",
            )

        executor.run()

    logger.info("\nAll dumps complete.")


if __name__ == "__main__":
    main()
