"""CPU import and frozen audit for the researcher's completed YouTube subset."""

import argparse
import logging
from pathlib import Path

from scripts.modal_youtube_recover import read, sha, write


def recovered_slots(manifest_path):
    """Check the complete recovery inventory, including zero-yield partitions."""
    manifest = read(manifest_path)
    root = manifest_path.parent
    catalog = read(root / "recovery-summary.json")
    downloaded = read(root / "returned/downloaded.json")
    workers = set(range(len(manifest["jobs"])))
    if (
        catalog.get("complete") is not True
        or catalog["manifest_sha256"] != sha(manifest_path)
        or downloaded["manifest_sha256"] != sha(manifest_path)
        or set(downloaded["workers"]) != workers
        or set(catalog["workers"]) != {str(i) for i in workers}
    ):
        raise ValueError("Incomplete recovery or recovery belongs to another handoff")
    slots = set()
    for worker in sorted(workers):
        path = root / "returned" / f"worker-{worker}.json"
        if sha(path) != catalog["workers"][str(worker)]["inventory_sha256"]:
            raise ValueError("Recovered worker inventory checksum changed")
        inventory = read(path)
        completed = inventory["completed_slots"]
        if (
            inventory["worker"] != worker
            or inventory["manifest_sha256"] != sha(manifest_path)
            or len(set(completed)) != len(completed)
            or not set(completed) <= set(manifest["jobs"][worker])
            or slots.intersection(completed)
        ):
            raise ValueError("Recovered partition allocation differs from handoff")
        slots.update(completed)
    return slots


def execute(manifest_path, output):
    from scripts.curation_audit import run as audit
    from scripts.curation_single import run
    from scripts.modal_youtube import common, transfer
    from scripts.youtube.download import _directory_lock

    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest, plan_path, plan = common.validate_manifest(manifest_path)
    # Keep all new state outside the source, handoff, and repository trees.
    for protected in (
        Path(plan["output_dir"]),
        Path(plan["input_dir"]),
        manifest_path.parent,
        Path(__file__).resolve().parents[2],
    ):
        protected = protected.resolve()
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError(
                "Finalization output must be separate from source and code"
            )
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        checkpoint = output / "import-audit.json"
        if checkpoint.exists():
            saved = read(checkpoint)
            if (
                saved.get("complete") is not True
                or saved["manifest_sha256"] != sha(manifest_path)
                or saved["plan_sha256"] != sha(plan_path)
                or saved["audit_summary_sha256"]
                != sha(output / "audit/review/summary.json")
            ):
                raise ValueError("Import/audit checkpoint changed")
            transfer.check_held(plan)
            # The finalizer rechecks every frozen worker/source fingerprint,
            # exports and lineage before using this completed audit. Avoid
            # rerunning the same two-pass million-file audit on every resume.
            logging.info(
                "Reusing completed import/audit; finalizer will verify bound files"
            )
            return read(output / "audit/review/summary.json")
        write(output / "progress.json", {"stage": "import", "complete": False})
        cloud = recovered_slots(manifest_path)
        transfer.merge(manifest_path)
        imported = read(manifest_path.parent / "import-summary.json")
        if not imported["complete"] or set(imported["imported_slots"]) != cloud:
            raise ValueError(
                "Import did not account for every recovered completed partition"
            )
        expected = cloud | set(manifest["completed_slots"])
        observed = {
            slot
            for slot in audit.source_slots(plan)
            if (run.folder(plan, slot) / "summary.json").exists()
            and read(run.folder(plan, slot) / "summary.json").get("workflow_finished")
        }
        if observed != expected:
            raise ValueError(
                "Completed YouTube scope differs from handoff plus recovery"
            )
        audit_dir = output / "audit"
        if (audit_dir / "frozen.json").exists() and {
            p["slot"] for p in read(audit_dir / "frozen.json")["parts"]
        } != expected:
            raise ValueError(
                "Existing audit freezes a different subset; use a new output directory"
            )
        write(output / "progress.json", {"stage": "audit", "complete": False})
        report = audit.execute(plan_path, audit_dir)
        if not report["audit_complete"] or report["completed_parts"] != len(expected):
            raise ValueError("Combined audit did not complete the expected subset")
        transfer.check_held(plan)
        write(
            output / "import-audit.json",
            {
                "complete": True,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha(manifest_path),
                "plan_path": str(plan_path),
                "plan_sha256": sha(plan_path),
                "recovered_slots": sorted(cloud),
                "completed_slots": sorted(expected),
                "unprocessed_slots": report["unprocessed_slots"],
                "audit_summary_sha256": sha(audit_dir / "review/summary.json"),
            },
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args.manifest, args.output)


if __name__ == "__main__":
    main()
