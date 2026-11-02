"""Audit all attempted YouTube partitions before the existing CPU finalizer."""

import argparse
import logging
from pathlib import Path

from scripts.curation_audit import run as audit
from scripts.curation_pool import run as pool
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_resume.run import read, require, verify_previous


def execute(path, manifest, previous, output):
    path, manifest, previous, output = (
        Path(p).resolve() for p in (path, manifest, previous, output)
    )
    require(
        verify_previous(previous, manifest, path),
        "Original recovery/finalization is incomplete",
    )
    plan = pool.single.load_plan(path)
    require(not pool.inventory(path, plan)[0], "Unfinished YouTube partitions remain")
    for protected in (
        path.parent,
        Path(plan["input_dir"]).resolve(),
        previous,
        Path(__file__).resolve().parents[2],
        manifest.parent,
    ):
        require(
            output != protected
            and not output.is_relative_to(protected)
            and not protected.is_relative_to(output),
            "Full-run output must be separate from inputs",
        )
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        checkpoint = output / "import-audit.json"
        if checkpoint.exists():
            saved = read(checkpoint)
            require(
                saved["complete"]
                and saved["plan_sha256"] == _sha256(path)
                and saved["manifest_sha256"] == _sha256(manifest)
                and saved["audit_summary_sha256"]
                == _sha256(output / "audit/review/summary.json"),
                "Full audit checkpoint changed",
            )
            # The existing finalizer validates every frozen source/worker
            # fingerprint, export, and lineage file again before selection.
            return read(output / "audit/review/summary.json")
        report = audit.execute(path, output / "audit")
        require(
            report["audit_complete"]
            and not report["unprocessed_slots"]
            and report["completed_parts"] == report["planned_parts"],
            "Full YouTube audit is incomplete",
        )
        frozen = read(output / "audit/frozen.json")
        _write_json(
            checkpoint,
            {
                "complete": True,
                "manifest_path": str(manifest),
                "manifest_sha256": _sha256(manifest),
                "plan_path": str(path),
                "plan_sha256": _sha256(path),
                "completed_slots": sorted(p["slot"] for p in frozen["parts"]),
                "unprocessed_slots": [],
                "audit_summary_sha256": _sha256(output / "audit/review/summary.json"),
                "prior_import_audit_sha256": _sha256(previous / "import-audit.json"),
                "scope": "Original recovered subset plus newly completed Marlowe partitions. Original selection policy unchanged; unresolved labels remain reported, not promoted.",
            },
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "manifest", "previous", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args.plan, args.manifest, args.previous, args.output)


if __name__ == "__main__":
    main()
