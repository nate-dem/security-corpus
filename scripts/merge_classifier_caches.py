"""Merge per-shard classifier caches into a single file ready for ingest.

Run this after all Slurm array tasks in run_youtube_classifier.slurm have
completed.  It reads every shard_XXXX.jsonl file from the cache directory,
deduplicates by video_id (last-write wins, which is safe since each video_id
appears in exactly one shard), and writes:

  data/youtube-transcripts/classifier_cache.jsonl
      The merged file consumed by ingest_youtube_transcripts.py --cache-path.

  data/youtube-transcripts/classifier_manifest.json
      Aggregate statistics: total classified, kept, rejected, retention rate,
      unique channels kept, estimated tokens, model name, timestamp.

Usage
-----
  python3 scripts/merge_classifier_caches.py

  # Custom paths:
  python3 scripts/merge_classifier_caches.py \
      --cache-dir data/youtube-transcripts/classifier_cache \
      --output    data/youtube-transcripts/classifier_cache.jsonl \
      --manifest  data/youtube-transcripts/classifier_manifest.json

  # Write an audit CSV with 20 kept and 20 rejected examples:
  python3 scripts/merge_classifier_caches.py --audit
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CACHE_DIR = Path("data/youtube-transcripts/classifier_cache")
DEFAULT_OUTPUT = Path("data/youtube-transcripts/classifier_cache.jsonl")
DEFAULT_MANIFEST = Path("data/youtube-transcripts/classifier_manifest.json")
OLD_ALLOWLIST = Path("data/youtube-transcripts/security_channels.txt")


def _load_old_allowlist() -> frozenset[str]:
    """Load the old channel-level allowlist if it still exists on disk.

    Used only for the comparison report — shows how many kept videos come from
    channels that would NOT have been included under the old 53-channel filter.
    """
    if not OLD_ALLOWLIST.exists():
        return frozenset()
    ids = {
        line.strip()
        for line in OLD_ALLOWLIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Write an audit CSV with 20 kept and 20 rejected examples for manual review.",
    )
    args = parser.parse_args()

    shard_files = sorted(args.cache_dir.glob("shard_*.jsonl"))
    if not shard_files:
        print(f"No shard_*.jsonl files found in {args.cache_dir}")
        print("Run sbatch scripts/run_youtube_classifier.slurm first.")
        return

    print(f"Merging {len(shard_files)} shard cache(s) from {args.cache_dir} ...")

    # --- Pass 1: read all entries, deduplicate by video_id ---
    # last-write wins; since each shard is processed by exactly one Slurm task
    # there should be no real duplicates, but cache reruns can produce them.
    all_entries: dict[str, dict] = {}
    models_seen: Counter = Counter()
    parse_errors = 0

    for shard_file in shard_files:
        with open(shard_file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    vid = entry.get("video_id")
                    if vid:
                        all_entries[vid] = entry
                        models_seen[entry.get("model", "unknown")] += 1
                except json.JSONDecodeError:
                    parse_errors += 1

    print(f"  {len(all_entries):,} unique video_ids loaded  ({parse_errors} malformed lines skipped)")

    # --- Pass 2: compute aggregate statistics ---
    kept_entries = [e for e in all_entries.values() if e.get("result", {}).get("should_keep", True)]
    rejected_entries = [e for e in all_entries.values() if not e.get("result", {}).get("should_keep", True)]

    total = len(all_entries)
    kept = len(kept_entries)
    rejected = len(rejected_entries)
    retention_rate = kept / total if total else 0.0

    # Relevance level breakdown across kept entries.
    level_counts: Counter = Counter(
        e["result"].get("relevance_level", "unknown")
        for e in kept_entries
    )

    # Unique channel_ids among kept entries.
    channels_kept = {e.get("channel_id") for e in kept_entries if e.get("channel_id")}

    # Estimated tokens: word_count * 1.3 is a rough tokens-per-word factor.
    est_tokens = sum(int((e.get("word_count") or 0) * 1.3) for e in kept_entries)

    # Comparison against the old 53-channel allowlist.
    old_allowlist = _load_old_allowlist()
    if old_allowlist:
        new_channels_not_in_old = channels_kept - old_allowlist
        old_channels_in_kept = channels_kept & old_allowlist
        print(f"\n  Comparison with old allowlist ({len(old_allowlist)} channels):")
        print(f"    Channels kept that WERE in old allowlist : {len(old_channels_in_kept):,}")
        print(f"    Channels kept that were NOT in old filter: {len(new_channels_not_in_old):,}  (net new coverage)")

    # --- Pass 3: write merged output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in all_entries.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  Written: {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")

    # --- Write manifest ---
    manifest = {
        "total_records_classified": total,
        "total_kept": kept,
        "total_rejected": rejected,
        "retention_rate": round(retention_rate, 4),
        "kept_high": level_counts.get("high", 0),
        "kept_medium": level_counts.get("medium", 0),
        "kept_low": level_counts.get("low", 0),
        "parse_failures_in_merge": parse_errors,
        "unique_channels_kept": len(channels_kept),
        "estimated_tokens_kept": est_tokens,
        "shards_merged": len(shard_files),
        "model": models_seen.most_common(1)[0][0] if models_seen else "unknown",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"  Written: {args.manifest}")

    # --- Print summary ---
    print(f"\n=== Merge complete ===")
    print(f"  Total classified : {total:,}")
    print(f"  Kept             : {kept:,}  ({retention_rate * 100:.1f}%)")
    print(f"    high           : {level_counts.get('high', 0):,}")
    print(f"    medium         : {level_counts.get('medium', 0):,}")
    print(f"    low            : {level_counts.get('low', 0):,}")
    print(f"  Rejected         : {rejected:,}")
    print(f"  Unique channels  : {len(channels_kept):,}")
    print(f"  Est. tokens      : ~{est_tokens:,}")

    # --- Optional audit CSV ---
    if args.audit:
        _write_audit_csv(kept_entries, rejected_entries, args.output.parent, old_allowlist)


def _write_audit_csv(
    kept: list[dict],
    rejected: list[dict],
    output_dir: Path,
    old_allowlist: frozenset[str],
) -> None:
    """Write 20 kept + 20 rejected examples for manual review.

    Kept entries are sorted by confidence ascending so the lowest-confidence
    decisions (most likely false positives) appear first.
    Rejected entries are sorted by confidence descending so borderline cases
    (confidence just above 0.70 threshold) appear first.
    """
    audit_path = output_dir / "audit_sample.csv"

    # Low confidence kept = most likely false positives — surface these first.
    kept_sorted = sorted(kept, key=lambda e: e.get("result", {}).get("confidence", 1.0))
    # High confidence rejected = clearest true negatives — also useful to spot false negatives.
    rejected_sorted = sorted(rejected, key=lambda e: e.get("result", {}).get("confidence", 0.0), reverse=True)

    sample = kept_sorted[:20] + rejected_sorted[:20]

    fieldnames = [
        "decision", "video_id", "title", "channel", "channel_id",
        "word_count", "relevance_level", "confidence", "reason",
        "topic_tags", "should_keep", "in_old_allowlist", "notes",
    ]

    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sample:
            result = entry.get("result", {})
            cid = entry.get("channel_id") or ""
            writer.writerow({
                "decision": "kept" if result.get("should_keep") else "rejected",
                "video_id": entry.get("video_id") or "",
                "title": entry.get("title") or "",
                "channel": entry.get("channel") or "",
                "channel_id": cid,
                "word_count": entry.get("word_count") or "",
                "relevance_level": result.get("relevance_level") or "",
                "confidence": result.get("confidence") or "",
                "reason": result.get("reason") or "",
                "topic_tags": "|".join(result.get("topic_tags") or []),
                "should_keep": result.get("should_keep"),
                "in_old_allowlist": cid in old_allowlist if old_allowlist else "n/a",
                "notes": "",
            })

    print(f"\n  Audit CSV written: {audit_path}  ({len(sample)} rows)")
    print(f"    First 20 rows: lowest-confidence kept (likely false positives)")
    print(f"    Last 20 rows : highest-confidence rejected (verify no false negatives)")


if __name__ == "__main__":
    main()
