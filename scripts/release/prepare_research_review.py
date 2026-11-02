"""Prepare reproducible, unreviewed samples from an audited corpus.

Uses the existing inspection budget of eight uniform and eight token-weighted
draws per source. These are review samples, never selection thresholds or a
claim that a human reviewed the corpus. Full documents are saved separately;
excerpt coverage is explicit. Nothing in the corpus is modified.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.inspect_final_corpus import SAMPLE_PER_SOURCE, excerpts
from scripts.release.normalize_final_schema import verify_parent

VERSION = "research-review-packet-v1"


def priorities(manifest_sha, identity, digest, tokens):
    # Midpoints of 52-bit bins stay strictly inside (0, 1).
    value = int(hashlib.sha256(
        (VERSION + manifest_sha + identity + digest).encode()
    ).hexdigest()[:13], 16)
    uniform = (value + 0.5) / 2**52
    final.require(tokens > 0, "Nonpositive token count")
    return {"uniform": uniform, "token_weighted": -math.log(uniform) / tokens}


def select(manifest, root, manifest_sha):
    heaps, populations = defaultdict(list), defaultdict(Counter)
    columns = ["source_id", "record_id", "content_hash", "content_length"]
    for ordinal, entry in enumerate(manifest["files"]):
        path = final.contained(root, entry["relative_path"])
        final.require(final.sha(path) == entry["sha256"], "Payload changed")
        offset = 0
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=2048, columns=columns):
                for i, row in enumerate(batch.to_pylist()):
                    source, tokens = row["source_id"], row["content_length"]
                    populations[source].update(records=1, tokens=tokens)
                    for category, priority in priorities(
                        manifest_sha, row["record_id"], row["content_hash"], tokens
                    ).items():
                        heap = heaps[source, category]
                        item = (-priority, ordinal, offset+i, row)
                        if len(heap) < SAMPLE_PER_SOURCE:
                            heapq.heappush(heap, item)
                        elif item[:3] > heap[0][:3]:
                            heapq.heapreplace(heap, item)
                offset += len(batch)
        final.require(offset == entry["records"], "File row count changed")
    selected = {}
    for (source, category), heap in sorted(heaps.items()):
        for neg, ordinal, row, proof in sorted(heap, reverse=True):
            note = selected.setdefault((ordinal, row), dict(proof, file=ordinal, row=row,
                row_index_basis="zero_based", manifest_sha256=manifest_sha, draws=[]))
            note["draws"].append(dict(category=category, priority=-neg))
    return selected, dict(populations)


def build(root, output):
    root, output = root.resolve(), output.resolve()
    final.require(not output.exists(), "Use a new packet directory")
    final.require(not output.is_relative_to(root), "Packet must be outside corpus")
    prepared, manifest = verify_parent(root)
    selected, populations = select(manifest, root, prepared["manifest_sha256"])
    final.require(sum(v["records"] for v in populations.values()) == manifest["totals"]["records"],
                  "Population count differs")
    output.mkdir(parents=True)
    (output / "documents").mkdir()
    shutil.copyfile(Path(__file__), output / "packet-code.py")
    by_file = defaultdict(dict)
    for (ordinal, row), note in selected.items():
        by_file[ordinal][row] = note
    seen = set()
    with (output / "review.jsonl").open("w") as dst:
        for ordinal, requests in sorted(by_file.items()):
            entry = manifest["files"][ordinal]
            path = final.contained(root, entry["relative_path"])
            final.require(final.sha(path) == entry["sha256"], "Selected payload changed")
            with pq.ParquetFile(path) as pf:
                offset = 0
                for batch in pf.iter_batches(batch_size=64):
                    wanted = [r for r in requests if offset <= r < offset+len(batch)]
                    for address in wanted:
                        row = batch.slice(address-offset, 1).to_pylist()[0]
                        proof = requests[address]
                        for key in ("source_id", "record_id", "content_hash", "content_length"):
                            final.require(proof[key] == row[key], "Selected identity changed")
                        text = row.pop("content")
                        final.require(hashlib.sha256(text.encode()).hexdigest() == proof["content_hash"],
                                      "Selected text hash differs")
                        relative = f"documents/{ordinal:06d}-{address:06d}.txt"
                        (output / relative).write_text(text, encoding="utf-8")
                        windows = excerpts(text)
                        note = dict(proof, relative_path=entry["relative_path"], document=relative,
                            document_sha256=final.sha(output / relative), metadata=row,
                            excerpts=windows, content_characters=len(text),
                            excerpt_characters=sum(len(x["text"]) for x in windows),
                            full_text_in_excerpts=sum(len(x["text"]) for x in windows)==len(text),
                            review_status="unreviewed", reviewer=None, substantive_usefulness=None,
                            extraction_complete=None, correctness_concerns=None,
                            within_existing_scope=None, notes=None, proposed_action=None)
                        dst.write(json.dumps(note, ensure_ascii=False, default=str)+"\n")
                        seen.add((ordinal, address))
                    offset += len(batch)
            final.require(final.sha(path) == entry["sha256"], "Payload changed during extraction")
    final.require(seen == set(selected), "Missing sample rows")
    report = dict(packet_complete=True, review_complete=False, version=VERSION,
        corpus=str(root), manifest_sha256=prepared["manifest_sha256"],
        unique_documents=len(selected), draws=sum(len(v["draws"]) for v in selected.values()),
        per_source_budget_per_sampling_method=SAMPLE_PER_SOURCE, populations=populations,
        method="Deterministic hash priorities; uniform and exponential-race token-weighted sampling without replacement within each source. Draws can overlap between methods.",
        limitations="Retained records only. Not a keep/drop audit or factual correctness test. Source strata are not corpus proportions; do not report unweighted sample percentages as a corpus error rate. Full text must be opened when excerpts are incomplete. Human labels and acceptance decisions are pending.",
        files_sha256={str(p.relative_to(output)):final.sha(p) for p in output.rglob("*") if p.is_file()})
    write(output / "summary.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = build(a.corpus, a.output)
    print(json.dumps({k:v for k,v in report.items() if k != "files_sha256"}, indent=2))
