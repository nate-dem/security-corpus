"""Extract unique channel_id + channel_name from all YouTube-Commons shards.

Downloads each shard one at a time, reads only the two channel columns
(skipping the large transcript text column), collects unique channels,
then deletes the shard to stay lean on disk.

Already-downloaded shards in data/youtube-transcripts/raw/ are read in-place
and not deleted.

Usage:
    python scripts/extract_youtube_channels.py

Output:
    data/youtube-transcripts/all_channels.csv  (channel_id, channel_name)
"""
import csv
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "PleIAs/YouTube-Commons"
RAW_DIR = Path("data/youtube-transcripts/raw")
OUT = Path("data/youtube-transcripts/all_channels.csv")


def list_shards() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def read_channel_cols(path: Path) -> dict[str, str]:
    table = pq.read_table(path, columns=["channel_id", "channel"])
    seen: dict[str, str] = {}
    for batch in table.to_batches():
        ids = batch.column("channel_id").to_pylist()
        names = batch.column("channel").to_pylist()
        for cid, name in zip(ids, names):
            if cid and cid not in seen:
                seen[cid] = name
    return seen


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shards = list_shards()
    print(f"Found {len(shards)} shards. Extracting channel columns…")

    all_channels: dict[str, str] = {}

    for i, shard in enumerate(shards, 1):
        local = RAW_DIR / shard
        tmp_path = None

        if local.exists():
            path = local
        else:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
                tmp.close()
                tmp_path = Path(tmp.name)
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=shard,
                    repo_type="dataset",
                    local_dir=str(tmp_path.parent),
                    local_dir_use_symlinks=False,
                )
                # hf_hub_download writes to local_dir/filename
                path = tmp_path.parent / shard
            except Exception as exc:
                print(f"  [{i}/{len(shards)}] SKIP {shard}: {exc}", file=sys.stderr)
                continue

        try:
            new = read_channel_cols(path)
            before = len(all_channels)
            all_channels.update(new)
            print(f"  [{i}/{len(shards)}] {shard} — {len(new):,} channels "
                  f"({len(all_channels) - before:,} new, {len(all_channels):,} total)")
        except Exception as exc:
            print(f"  [{i}/{len(shards)}] ERROR {shard}: {exc}", file=sys.stderr)
        finally:
            if tmp_path is not None:
                # clean up temp download (not the pre-existing RAW_DIR shard)
                downloaded = tmp_path.parent / shard
                downloaded.unlink(missing_ok=True)

    print(f"\nDone. {len(all_channels):,} unique channels across all shards.")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel_id", "channel_name"])
        for cid, name in sorted(all_channels.items()):
            w.writerow([cid, name])

    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
