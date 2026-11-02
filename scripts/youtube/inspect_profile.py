#!/usr/bin/env python3
"""Verify copied profile artifacts and measure the diagnostic sample locally.

This does not estimate whole-corpus token totals or select training records.
The optional language export is an unlabeled pilot input, not a retained corpus.
Uses the main project's offline, checksum-verified cl100k_base vocabulary.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.utils import compute_content_hash, compute_token_count  # noqa: E402


def read_verified_sample(root: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("complete") is not True:
        raise ValueError("Profile is incomplete")
    sample_bytes = (root / "inspection_sample.jsonl").read_bytes()
    if hashlib.sha256(sample_bytes).hexdigest() != summary["inspection_sample"]["sha256"]:
        raise ValueError("Inspection sample checksum mismatch")
    rows = [json.loads(line) for line in sample_bytes.splitlines()]
    if len(rows) != summary["inspection_sample"]["records"]:
        raise ValueError("Inspection sample row count mismatch")
    shards = {row["shard"] for row in summary["shard_schemas"]}
    locations = set()
    for row in rows:
        location = (row["source_shard"], row["source_row"])
        if location in locations or row["source_shard"] not in shards:
            raise ValueError("Duplicate sample location or unknown shard")
        locations.add(location)
        if row["dataset_revision"] != summary["dataset_revision"]:
            raise ValueError("Sample revision mismatch")
        if not 0 <= row["source_row"] < row["shard_rows"]:
            raise ValueError("Sample source row out of bounds")
        if not 0 < row["selection_probability"] <= 1:
            raise ValueError("Invalid sample selection probability")
        if row["text"] is not None and not isinstance(row["text"], str):
            raise ValueError("Unexpected sample text type")
    return summary, rows


def _quantiles(values: list[int]) -> dict:
    ordered = sorted(values)
    def percentile(p):
        if not ordered:
            return None
        position = p * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"p50": percentile(.5), "p90": percentile(.9), "p99": percentile(.99),
            "max": max(ordered) if ordered else None}


def analyze(rows: list[dict]) -> tuple[dict, list[dict]]:
    features = []
    by_language = defaultdict(list)
    for index, row in enumerate(rows):
        text = row["text"]
        feature = {"sample_index": index, "source_shard": row["source_shard"],
                   "source_row": row["source_row"], "video_id": row["video_id"],
                   "transcription_language": row["transcription_language"],
                   "original_language": row["original_language"],
                   "content_sha256": compute_content_hash(text) if text is not None else None,
                   "cl100k_base_tokens": compute_token_count(text) if text is not None else None,
                   "actual_characters": len(text) if text is not None else None,
                   "blank_text": text is None or not text.strip(),
                   "reported_characters_match": text is not None and len(text) == row["character_count"],
                   "selection_probability": row["selection_probability"]}
        features.append(feature)
        by_language[row["transcription_language"]].append(feature)
    nonempty_hashes = Counter(r["content_sha256"] for r in features if not r["blank_text"])
    stats = {"records": len(rows), "blank_text_records": sum(r["blank_text"] for r in features),
             "reported_character_count_mismatches": sum(not r["reported_characters_match"] for r in features),
             "nonempty_exact_text_duplicate_extra_rows": sum(n - 1 for n in nonempty_hashes.values()),
             "cl100k_base_tokens": sum(r["cl100k_base_tokens"] or 0 for r in features),
             "token_quantiles": _quantiles([r["cl100k_base_tokens"] for r in features
                                            if r["cl100k_base_tokens"] is not None]),
             "by_language": []}
    for language, group in sorted(by_language.items(), key=lambda x: str(x[0])):
        stats["by_language"].append({
            "language": language, "records": len(group),
            "blank_text_records": sum(r["blank_text"] for r in group),
            "cl100k_base_tokens": sum(r["cl100k_base_tokens"] or 0 for r in group),
            "token_quantiles": _quantiles([r["cl100k_base_tokens"] for r in group
                                           if r["cl100k_base_tokens"] is not None]),
        })
    return stats, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-language", help="Exact metadata label for a diagnostic export; no corpus filtering")
    args = parser.parse_args()
    if args.profile_dir.resolve() == args.output_dir.resolve():
        parser.error("Keep review outputs separate from the copied profile")
    summary, rows = read_verified_sample(args.profile_dir)
    stats, features = analyze(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "dataset_revision": summary["dataset_revision"],
              "profile_summary_sha256": hashlib.sha256((args.profile_dir / "summary.json").read_bytes()).hexdigest(),
              "inspection_sample_sha256": summary["inspection_sample"]["sha256"],
              "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sample_checksum_verified": True, "sample": stats,
              "limitations": "Sample measurements only; no whole-corpus token estimate, semantic labels, or keep/drop decisions.",
              "pilot_language_label": args.pilot_language}
    for name, data in (("sample_features.jsonl", features),
                       ("pilot_available.jsonl", [dict(row, sample_analysis=features[i])
                                                  for i, row in enumerate(rows)
                                                  if args.pilot_language is not None
                                                  and row["transcription_language"] == args.pilot_language])):
        (args.output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
    (args.output_dir / "sample_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
