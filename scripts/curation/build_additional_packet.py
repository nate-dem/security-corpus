"""Sample existing diagnostic pools, excluding the reviewed documents.

This is additional development material, not an independent population sample.
Raw texts remain unchanged, complete, and bound to their source locations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.web_corpora.profile import text_field
from scripts.youtube_filter.rubric import LABELS
from .score import load_packet


def sample_rows(rows, source, excluded, count, seed):
    rng = random.Random(f"{seed}:{source}")
    seen = set()
    sample = []
    eligible = 0
    for row in rows:
        if source == "youtube-commons":
            text = row["text"]
            features = row
            lineage = {
                k: row[k] for k in ("dataset_revision", "content_hash", "sample_arms")
            }
            lineage["source_locations"] = row["locations"]
            key = "youtube-additional:" + row["content_hash"]
        else:
            features = row["features"]
            if features["text_state"] != "valid":
                continue
            text = row["raw_record"][text_field(source)]
            lineage = {
                k: row[k]
                for k in (
                    "dataset_revision",
                    "source_shard",
                    "source_row",
                    "selection_probability",
                )
            }
            lineage["content_hash"] = features["content_hash"]
            lineage["url"] = features.get("url")
            key = f"{source}:{row['source_shard']}:{row['source_row']}"
        digest = compute_content_hash(text)
        if digest != features["content_hash"]:
            raise ValueError("Diagnostic text hash mismatch")
        if digest in excluded or digest in seen:
            continue
        seen.add(digest)
        eligible += 1
        slot = len(sample) if len(sample) < count else rng.randrange(eligible)
        if slot < count:
            case = {
                "case_id": key,
                "source": source,
                "text": text,
                "text_sha256": digest,
                "content_length": features["content_length"],
                "unit": "complete_document",
                "purpose": "development_only",
                "lineage": lineage,
                "sampling_note": "Reservoir from an existing diagnostic pool, after document-hash exclusion. Not a population/yield sample.",
            }
            if slot == len(sample):
                sample.append(case)
            else:
                sample[slot] = case
    for case in sample:
        if compute_token_count(case["text"]) != case["content_length"]:
            raise ValueError("Selected diagnostic token count mismatch")
    return sorted(sample, key=lambda c: c["case_id"]), eligible


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument(
        "--per-source",
        type=int,
        default=100,
        help="Diagnostic review workload, not a content threshold",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.per_source < 1:
        p.error("per-source must be positive")
    prior_path = a.project / "reports/curation/development-v2/packet.json"
    prior = load_packet(prior_path)
    excluded = {c["text_sha256"] for c in prior["cases"]} | {
        c["lineage"].get("content_hash") for c in prior["cases"]
    }
    paths = {
        "youtube-commons": a.project / "reports/youtube/pilot-v1/candidates.jsonl",
        "redsage-cfw": a.project
        / "reports/web-corpora/redsage-cfw/profile-v1/inspection_sample.jsonl",
        "primus-fineweb": a.project
        / "reports/web-corpora/primus-fineweb/profile-v2/inspection_sample.jsonl",
    }
    cases = []
    pools = {}
    for source, path in paths.items():
        with path.open() as f:
            selected, eligible = sample_rows(
                map(json.loads, f), source, excluded, a.per_source, a.seed
            )
        if len(selected) != a.per_source:
            raise ValueError(
                f"Insufficient additional candidates for {source}: {len(selected)}"
            )
        cases.extend(selected)
        pools[source] = {
            "path": str(path.relative_to(a.project)),
            "sha256": _sha256(path),
            "eligible_pool_texts": eligible,
            "sampled": len(selected),
        }
    packet = {
        "version": 1,
        "purpose": "development_only",
        "labels": LABELS,
        "cases": cases,
        "prior_packet_sha256": prior["packet_sha256"],
        "sampling": {"seed": a.seed, "per_source": a.per_source, "pools": pools},
        "independent_ground_truth": False,
        "scope": "Additional diagnostic texts excluded from the 89 reviewed documents; pools have earlier pipeline exposure. No yield or independent accuracy claim.",
    }
    packet["packet_sha256"] = hashlib.sha256(
        json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    target = a.output_dir / "packet.json"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and json.loads(target.read_text()) != packet:
        raise ValueError("Packet changed; use a new directory")
    _write_json(target, packet)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "tokens": sum(c["content_length"] for c in cases),
                "pools": pools,
                "packet_sha256": packet["packet_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
