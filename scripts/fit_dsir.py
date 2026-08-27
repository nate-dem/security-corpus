#!/usr/bin/env python3
"""Fit a DSIR-style security-domain scorer for FineWeb filtering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest.connectors.web import docs_from_input, fineweb_record_text, fit_dsir_scorer


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    positive_globs = args.positive_glob or [
        "final-data/normalized/source_id=*/*.parquet",
        "final-data/qwen_qa_kept/source_id=*/*.parquet",
        "final-data/academic_papers/*.parquet",
    ]
    background_docs = docs_from_input(
        input_glob=args.input_glob,
        fineweb_dataset=args.fineweb_dataset,
        fineweb_config=args.fineweb_config,
        split=args.split,
    )
    background_texts = _texts_from_docs(background_docs, limit=args.sample_size)

    from ingest.connectors.web.fineweb import iter_positive_texts

    scorer = fit_dsir_scorer(
        iter_positive_texts(positive_globs, limit=args.positive_limit or args.sample_size),
        background_texts,
        positive_limit=args.positive_limit or args.sample_size,
        background_limit=args.sample_size,
        ngram_range=(args.min_ngram, args.max_ngram),
        min_feature_count=args.min_feature_count,
        alpha=args.alpha,
        metadata={
            "positive_globs": positive_globs,
            "fineweb_dataset": args.fineweb_dataset,
            "fineweb_config": args.fineweb_config,
            "split": args.split,
            "input_glob": args.input_glob,
            "sample_size": args.sample_size,
        },
    )
    scorer.save(args.output)
    _write_reports(args.report_dir, args.output, scorer)
    print(f"Wrote scorer: {args.output}")
    print(f"Positive docs: {scorer.metadata['positive_docs']:,}")
    print(f"Background docs: {scorer.metadata['background_docs']:,}")
    print(f"Features: {scorer.metadata['vocab_size']:,}")
    return 0


def _texts_from_docs(docs, *, limit: int):
    count = 0
    for doc in docs:
        text = fineweb_record_text(doc)
        if text:
            yield text
            count += 1
            if count >= limit:
                return


def _write_reports(report_dir: Path, output: Path, scorer) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scorer_path": str(output),
        "metadata": scorer.metadata,
        "top_positive_features": sorted(scorer.weights.items(), key=lambda item: item[1], reverse=True)[:50],
        "top_background_features": sorted(scorer.weights.items(), key=lambda item: item[1])[:50],
    }
    (report_dir / "dsir_fit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# FineWeb DSIR Fit Summary",
        "",
        f"- Scorer: `{output}`",
        f"- Version: `{scorer.metadata.get('version')}`",
        f"- Positive docs: {scorer.metadata.get('positive_docs'):,}",
        f"- Background docs: {scorer.metadata.get('background_docs'):,}",
        f"- Features: {scorer.metadata.get('vocab_size'):,}",
        "",
        "## Top Security-Weighted Features",
        "",
    ]
    for term, weight in payload["top_positive_features"][:25]:
        lines.append(f"- `{term}`: {weight:.4f}")
    lines.extend(["", "## Top Background-Weighted Features", ""])
    for term, weight in payload["top_background_features"][:25]:
        lines.append(f"- `{term}`: {weight:.4f}")
    (report_dir / "dsir_fit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-glob", action="append", help="Parquet glob(s) for positive security corpus text.")
    parser.add_argument("--sample-size", type=int, default=1_000_000, help="Number of raw FineWeb background docs to sample.")
    parser.add_argument("--positive-limit", type=int, help="Limit positive docs; defaults to sample-size.")
    parser.add_argument("--fineweb-dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--fineweb-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-glob", help="Use local FineWeb Parquet/JSONL files instead of Hugging Face streaming.")
    parser.add_argument("--output", type=Path, default=Path("data/fineweb/dsir_scorer.pkl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/fineweb"))
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-feature-count", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
