"""Classify YouTube-Commons videos as cybersecurity-relevant using vLLM.

This script is designed to run as a Slurm array job on Marlowe HPC, with one
task per parquet shard.  It can also be run locally with --sample N for
piloting and prompt inspection.

What it does
------------
For a given shard (cctube_N.parquet):

  1. Download the shard from HuggingFace if it is not already on disk.
  2. Read only the metadata columns (no transcript text) — pass 1.
  3. Apply the language + word count pre-gates.  Records that fail these
     gates are never sent to the classifier (no point paying inference cost
     on non-English or sub-50-word transcripts).
  4. Load the existing per-shard cache and skip any video_id already classified.
  5. Build a batch of chat-format prompts for the remaining candidates.
  6. Run vLLM batch inference (all prompts in one call; vLLM handles scheduling).
  7. Parse each response with the three-level fallback chain in youtube_classifier.
  8. Append new results to the per-shard JSONL cache.
  9. Print a summary of what was kept, rejected, and any parse failures.

Cache location
--------------
  data/youtube-transcripts/classifier_cache/shard_{N:04d}.jsonl

After all 439 array tasks finish, run scripts/merge_classifier_caches.py to
combine them into a single classifier_cache.jsonl ready for ingest.

Usage
-----
On Marlowe (submitted via run_youtube_classifier.slurm):
  SLURM_ARRAY_TASK_ID is read automatically — no --shard flag needed.

Local pilot (shard 0, first 200 records):
  python3 scripts/classify_youtube_videos.py --shard 0 --sample 200

Local dry-run (inspect prompts without calling the model):
  python3 scripts/classify_youtube_videos.py --shard 0 --sample 10 --dry-run

Dependencies
------------
  pip install -e ".[youtube]"   # installs vllm + huggingface_hub
  vllm requires CUDA — run on GPU nodes only.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

# youtube_classifier is safe to import everywhere (no vllm import at top level).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ingest.connectors.youtube_classifier import (
    SYSTEM_PROMPT,
    _build_user_message,
    _parse_response,
    _LENIENT_FALLBACK,
)
from ingest.connectors.youtube_transcripts import MIN_WORD_COUNT, _GATE_COLS, _is_english

REPO_ID = "PleIAs/YouTube-Commons"
DEFAULT_RAW_DIR = Path("data/youtube-transcripts/raw")
DEFAULT_CACHE_DIR = Path("data/youtube-transcripts/classifier_cache")
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Description enrichment
# ---------------------------------------------------------------------------

def _load_descriptions(descriptions_path: Path) -> dict[str, str]:
    """Load video_id → description from Rijgersberg/YouTube-Commons-descriptions.

    The dataset is structured as a parquet directory (one or more .parquet files).
    We read only the video_id and description columns, then build a plain dict
    for O(1) lookup per candidate.  The dict lives in memory for the lifetime of
    the Slurm task; at ~2–4 GB uncompressed it fits comfortably within the 100 GB
    memory allocation.

    Returns an empty dict when the path doesn't exist or has no parquet files —
    the caller falls back to the 5-field prompt without a description line.
    """
    if not descriptions_path.exists():
        print(f"  Descriptions path not found: {descriptions_path} — skipping enrichment")
        return {}

    desc_map: dict[str, str] = {}
    parquet_files = sorted(descriptions_path.glob("**/*.parquet"))
    if not parquet_files:
        print(f"  No parquet files in {descriptions_path} — skipping enrichment")
        return {}

    for pf_path in parquet_files:
        try:
            table = pq.read_table(pf_path, columns=["video_id", "description"])
            for vid, desc in zip(
                table.column("video_id").to_pylist(),
                table.column("description").to_pylist(),
            ):
                if vid and desc:
                    desc_map[vid] = desc
        except Exception as exc:
            print(f"  WARNING: could not read {pf_path}: {exc}", file=sys.stderr)
            continue

    print(f"  Loaded descriptions for {len(desc_map):,} videos")
    return desc_map


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_shard_cache(cache_path: Path) -> set[str]:
    """Return the set of video_ids already classified in this shard's cache."""
    if not cache_path.exists():
        return set()
    seen: set[str] = set()
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                vid = entry.get("video_id")
                if vid:
                    seen.add(vid)
            except json.JSONDecodeError:
                pass
    return seen


def _append_cache(cache_path: Path, entries: list[dict]) -> None:
    """Append a batch of classification results to the per-shard JSONL cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Shard download
# ---------------------------------------------------------------------------

def _ensure_shard(shard_index: int, raw_dir: Path) -> Path | None:
    """Return the local path to the shard, downloading it if necessary."""
    filename = f"cctube_{shard_index}.parquet"
    local_path = raw_dir / filename
    if local_path.exists():
        return local_path

    print(f"  Shard {shard_index} not on disk — downloading from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        raw_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"  Downloaded: {local_path}")
        return local_path
    except Exception as exc:
        print(f"  ERROR: could not download shard {shard_index}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _read_candidates(shard_path: Path, already_seen: set[str]) -> list[dict]:
    """Read metadata columns from the shard and return records that need classifying.

    A record is a candidate when:
      - It passes the language gate (_is_english)
      - It passes the word count gate (>= MIN_WORD_COUNT)
      - Its video_id is not already in the per-shard cache

    The text column is never read here — only metadata.
    """
    try:
        pf = pq.ParquetFile(shard_path)
    except Exception as exc:
        print(f"  ERROR: cannot open {shard_path}: {exc}", file=sys.stderr)
        return []

    candidates: list[dict] = []
    total_rows = 0

    for rg in range(pf.metadata.num_row_groups):
        try:
            table = pf.read_row_group(rg, columns=_GATE_COLS)
        except Exception:
            continue

        col_data = {name: table.column(name).to_pylist() for name in table.schema.names}
        n = table.num_rows
        total_rows += n

        for i in range(n):
            record = {col: col_data[col][i] for col in col_data}
            if not _is_english(record):
                continue
            if (record.get("word_count") or 0) < MIN_WORD_COUNT:
                continue
            vid = record.get("video_id") or ""
            if vid in already_seen:
                continue
            candidates.append(record)

    print(f"  Total rows: {total_rows:,}  |  Candidates after pre-gates: {len(candidates):,}  |  Cache hits skipped: {len(already_seen):,}")
    return candidates


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def _load_model(model_name: str):
    """Load the vLLM model and return (llm, sampling_params, tokenizer).

    Called once per Slurm task.  prefix_caching=True means the constant system
    prompt's KV state is computed once and reused for every record in the
    shard, cutting prefill cost by ~80%.
    """
    # vllm is imported here (not at module top) so this file can be imported
    # on macOS dev machines where vllm is not installed.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"  Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=2048,
        enable_prefix_caching=True,  # reuse KV for constant system prompt
    )
    # temperature=0 → deterministic/greedy; reproducible across reruns of the same shard.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
    )
    # Tokenizer is used to apply the chat template and build the final prompt strings.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Model loaded.")
    return llm, sampling_params, tokenizer


def _build_prompts(
    records: list[dict],
    tokenizer,
    desc_map: dict[str, str],
) -> list[str]:
    """Convert metadata records into chat-formatted prompt strings for vLLM.

    Each prompt contains the constant system message plus a per-record user
    message.  When desc_map is non-empty, the video description is appended
    (truncated to 300 chars) for records whose video_id appears in the map.
    Records without a description entry use the 5-field fallback.

    For Llama (and Qwen3), enable_thinking=False is attempted to suppress
    chain-of-thought output that adds latency with no quality benefit for
    structured JSON responses.  Falls back gracefully for models that don't
    support the kwarg.
    """
    prompts = []
    for record in records:
        vid = record.get("video_id") or ""
        description = desc_map.get(vid)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(record, description)},
        ]
        # enable_thinking=False is a qwen3-specific chat template kwarg.
        # Falls back gracefully for other models that don't support it.
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Model tokenizer doesn't accept enable_thinking — just apply normally.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def _run_inference(
    records: list[dict],
    llm,
    sampling_params,
    tokenizer,
    model_name: str,
    cache_path: Path,
    batch_size: int,
    desc_map: dict[str, str],
) -> dict:
    """Run vLLM inference on all candidates in batches and write cache entries.

    Returns a dict of summary counts for the caller to print.
    """
    counts = {"kept": 0, "rejected": 0, "parse_failures": 0,
              "high": 0, "medium": 0, "low": 0, "not_relevant": 0}
    classified_at = datetime.now(timezone.utc).isoformat()

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        prompts = _build_prompts(batch, tokenizer, desc_map)

        # vLLM processes the entire batch in one call with continuous batching.
        outputs = llm.generate(prompts, sampling_params)

        cache_entries = []
        for record, output in zip(batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            result = _parse_response(raw_text)

            if result is None:
                counts["parse_failures"] += 1
                result = _LENIENT_FALLBACK

            if result.should_keep:
                counts["kept"] += 1
                counts[result.relevance_level] = counts.get(result.relevance_level, 0) + 1
            else:
                counts["rejected"] += 1

            cache_entries.append({
                "video_id": record.get("video_id") or "",
                "channel_id": record.get("channel_id") or "",
                "channel": record.get("channel") or "",
                "title": record.get("title") or "",
                "word_count": record.get("word_count") or 0,
                "result": result.model_dump(),
                "classified_at": classified_at,
                "model": model_name,
            })

        _append_cache(cache_path, cache_entries)

        done = min(batch_start + batch_size, len(records))
        print(f"    Batch {batch_start}–{done - 1}: {sum(1 for e in cache_entries if e['result']['should_keep'])} kept / {len(cache_entries)} classified")

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard index (0–438).  Defaults to $SLURM_ARRAY_TASK_ID when running under Slurm.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help=f"Raw parquet directory (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"Per-shard cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Records per vLLM call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--sample", type=int, default=None, metavar="N", help="Classify only the first N candidates (for local pilots)")
    parser.add_argument("--dry-run", action="store_true", help="Print the first 5 prompts and exit without calling the model")
    parser.add_argument("--delete-after", action="store_true", help="Delete the raw shard after classification to save scratch space")
    parser.add_argument(
        "--descriptions-path",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to Rijgersberg/YouTube-Commons-descriptions parquet directory. "
             "When provided, video descriptions are added to classifier prompts (~300 chars). "
             "Videos without an entry in the descriptions dataset use the 5-field fallback.",
    )
    args = parser.parse_args()

    # Resolve shard index: command-line flag overrides SLURM_ARRAY_TASK_ID.
    shard_index = args.shard
    if shard_index is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_id is None:
            parser.error("--shard is required when not running under Slurm (SLURM_ARRAY_TASK_ID not set)")
        shard_index = int(slurm_id)

    cache_path = args.cache_dir / f"shard_{shard_index:04d}.jsonl"

    print(f"\n=== Shard {shard_index} ===")
    print(f"  Cache: {cache_path}")
    t0 = time.time()

    # Step 1: ensure shard is on disk.
    shard_path = _ensure_shard(shard_index, args.raw_dir)
    if shard_path is None:
        sys.exit(1)

    # Step 2: load existing cache to skip already-classified records.
    already_seen = _load_shard_cache(cache_path)

    # Step 3: read metadata and filter to unclassified candidates.
    candidates = _read_candidates(shard_path, already_seen)

    if args.sample is not None:
        candidates = candidates[: args.sample]
        print(f"  --sample: limiting to first {len(candidates)} candidates")

    if not candidates:
        print("  Nothing to classify (all records already in cache or no candidates passed pre-gates).")
        if args.delete_after:
            shard_path.unlink(missing_ok=True)
        return

    # Step 4: load description enrichment map (optional).
    # desc_map is a dict[video_id, description] built once per task and shared
    # across all batches.  Empty dict when --descriptions-path is not provided.
    desc_map: dict[str, str] = {}
    if args.descriptions_path is not None:
        desc_map = _load_descriptions(args.descriptions_path)
        enriched = sum(1 for r in candidates if r.get("video_id") in desc_map)
        print(f"  Description coverage: {enriched:,} / {len(candidates):,} candidates ({enriched / len(candidates) * 100:.1f}%)")

    # Step 5: dry-run — print the system + user messages in plain text and exit.
    # The tokenizer (and vllm) are not imported in dry-run mode so this works
    # on any machine, including macOS dev boxes without CUDA.
    if args.dry_run:
        sample = candidates[:5]
        print(f"\n--- Dry-run: first {len(sample)} prompt(s) (untokenized) ---")
        print(f"[SYSTEM]\n{SYSTEM_PROMPT}\n")
        for i, record in enumerate(sample):
            vid = record.get("video_id") or ""
            user_msg = _build_user_message(record, desc_map.get(vid))
            print(f"[User message {i}]\n{user_msg}\n{'—' * 60}")
        return

    # Step 6: load model (once per Slurm task).
    llm, sampling_params, tokenizer = _load_model(args.model)

    # Step 7: classify in batches, write cache incrementally.
    print(f"  Classifying {len(candidates):,} candidates in batches of {args.batch_size}...")
    counts = _run_inference(
        candidates, llm, sampling_params, tokenizer,
        args.model, cache_path, args.batch_size,
        desc_map=desc_map,
    )

    # Step 7: print shard summary.
    elapsed = time.time() - t0
    total_classified = counts["kept"] + counts["rejected"]
    retention = counts["kept"] / total_classified * 100 if total_classified else 0
    est_tokens = sum(
        int((r.get("word_count") or 0) * 1.3)
        for r in candidates
    )

    print(f"\n--- Shard {shard_index} complete ---")
    print(f"  Classified    : {total_classified:,}")
    print(f"  Kept          : {counts['kept']:,}  ({retention:.1f}%)")
    print(f"    high        : {counts.get('high', 0):,}")
    print(f"    medium      : {counts.get('medium', 0):,}")
    print(f"    low         : {counts.get('low', 0):,}")
    print(f"    not_relevant (low-conf): {counts.get('not_relevant', 0):,}")
    print(f"  Rejected      : {counts['rejected']:,}")
    print(f"  Parse failures: {counts['parse_failures']:,}  (kept by lenient fallback)")
    print(f"  Est. tokens   : ~{est_tokens:,}  (word_count * 1.3 for kept records)")
    print(f"  Wall time     : {elapsed / 60:.1f} min")

    # Step 8: optionally remove shard to save scratch space.
    if args.delete_after:
        shard_path.unlink(missing_ok=True)
        print(f"  Deleted raw shard: {shard_path}")


if __name__ == "__main__":
    main()
