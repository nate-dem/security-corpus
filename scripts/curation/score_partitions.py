"""Score explicitly selected prepared partitions; no automatic release selection.

Keeps one model resident across work units. Inputs and text remain unchanged;
outputs are provenance-bound diagnostic sidecars for subsequent audited assembly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter.rubric import LABELS
from . import score
from .runtime import ResidentModel


def load_prepared(root):
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary.get("complete") is not True
        or summary.get("version") != "curation-inputs-v1"
        or summary["run_config_sha256"] != _sha256(root / "run-config.json")
    ):
        raise ValueError("A completed, bound preparation manifest is required")
    seen = set()
    for part in summary["parts"]:
        path = (root / part["path"]).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or path in seen
            or part["records"] <= 0
            or part["tokens"] <= 0
        ):
            raise ValueError("Invalid or repeated preparation part")
        seen.add(path)
    if (
        sum(p["records"] for p in summary["parts"]) != summary["records"]
        or sum(p["tokens"] for p in summary["parts"]) != summary["tokens"]
    ):
        raise ValueError("Preparation manifest totals disagree")
    return summary


def part_packet(root, part, manifest_sha):
    path = (root / part["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or _sha256(path) != part["sha256"]:
        raise ValueError("Prepared part path/checksum mismatch")
    cases = []
    with pq.ParquetFile(path) as f:
        for batch in f.iter_batches(batch_size=16, use_threads=False):
            for row in batch.to_pylist():
                if row["kind"] != (
                    "youtube" if row["source"] == "youtube-commons" else "web"
                ):
                    raise ValueError("Prepared source kind mismatch")
                cases.append(
                    {
                        "case_id": f"{row['source']}:{row['source_shard']}:{row['source_row']}",
                        "source": row["source"],
                        "text": row["text"],
                        "text_sha256": row["content_hash"],
                        "content_length": row["content_length"],
                        "unit": "complete_document",
                        "purpose": "candidate_scoring",
                        "lineage": {
                            k: row[k]
                            for k in (
                                "dataset_revision",
                                "source_shard",
                                "source_row",
                                "source_copies",
                            )
                        },
                    }
                )
    if (
        len(cases) != part["records"]
        or sum(c["content_length"] for c in cases) != part["tokens"]
    ):
        raise ValueError("Prepared part counts mismatch")
    packet = {
        "version": 1,
        "purpose": "candidate_scoring",
        "labels": LABELS,
        "cases": cases,
        "prepared_manifest_sha256": manifest_sha,
        "prepared_part": part,
        "selection_decision": None,
    }
    packet["packet_sha256"] = hashlib.sha256(
        json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return packet


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--part-index",
        type=int,
        action="append",
        required=True,
        help="Explicit zero-based indices from the completed manifest",
    )
    p.add_argument("--rubric-version", choices=("v3", "critic-v1"), required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)
    root = a.input_dir.resolve()
    output = a.output_dir.resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        p.error("Input and output directories must be separate")
    summary = load_prepared(root)
    digest = _sha256(root / "summary.json")
    if len(set(a.part_index)) != len(a.part_index) or any(
        i < 0 or i >= len(summary["parts"]) for i in a.part_index
    ):
        p.error("Part indices must be distinct and present in the manifest")
    packets = output / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    factory = ResidentModel()
    failed = False
    size = 8 if a.rubric_version == "v3" else 32
    revision = (
        "b968826d9c46dd6066d109eabc6255188de91218"
        if size == 8
        else "9216db5781bf21249d130ec9da846c4624c16137"
    )
    for i in a.part_index:
        packet = part_packet(root, summary["parts"][i], digest)
        packet_path = packets / f"{i:06d}.json"
        if packet_path.exists() and json.loads(packet_path.read_text()) != packet:
            raise ValueError("Part binding changed; choose a new output directory")
        _write_json(packet_path, packet)
        args = [
            "--packet",
            str(packet_path),
            "--output-dir",
            str(output / "parts" / f"{i:06d}"),
            "--rubric-version",
            a.rubric_version,
            "--model",
            f"Qwen/Qwen3-{size}B",
            "--model-revision",
            revision,
            "--tensor-parallel-size",
            "1" if size == 8 else "2",
        ]
        if a.dry_run:
            args.append("--dry-run")
        if a.local_files_only:
            args.append("--local-files-only")
        status = score.main(args, model_factory=factory)
        if status not in (0, 2):
            return status
        failed |= status == 2
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
