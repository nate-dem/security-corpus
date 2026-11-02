"""Freeze and finalize the researcher's completed YouTube subset, CPU only.

This is deliberately independent of the GPU supervisor. Frozen audit/finalizer
code and eligibility/deduplication policies are reused without modification.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess

from scripts.curation_audit import run as audit
from scripts.curation_pool import run as pool
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_cpu_io.run import parallel_fingerprints

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
PROJECT = Path(os.environ.get("YT_SNAPSHOT_PROJECT", str(TOOLS))).resolve()
LOG = logging.getLogger(__name__)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity():
    paths = [*HERE.glob("*.py"), *HERE.glob("*.sh"), *HERE.glob("*.sbatch")]
    paths += [TOOLS / "scripts/youtube_cpu_io/run.py"]
    paths += [
        TOOLS / "scripts/release" / name
        for name in ("record_rights.py", "rights_sources.json", "audit_snapshot.py")
    ]
    return {str(p.relative_to(TOOLS)): _sha256(p) for p in sorted(paths)}


def validate_output(output, path, plan, web):
    for protected in (PROJECT, TOOLS, path.parent, Path(plan["input_dir"]), web):
        protected = protected.resolve()
        require(
            output != protected
            and not output.is_relative_to(protected)
            and not protected.is_relative_to(output),
            "Snapshot output must be separate from code and existing inputs",
        )


@contextmanager
def existing_source_lock(root):
    """Lock the existing inode without creating or writing anything on scratch."""
    path = root / ".download.lock"
    try:
        # O_RDWR supports exclusive flock on NFS; no O_CREAT and no writes.
        descriptor = os.open(path, os.O_RDWR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Expected source lock is missing: {path}; no file was created"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def source_locks_without_creation(plan):
    original = audit._directory_lock
    root = Path(plan["output_dir"]).resolve()

    def lock(directory):
        if Path(directory).resolve().is_relative_to(root):
            return existing_source_lock(directory)
        return original(directory)

    audit._directory_lock = lock
    try:
        yield
    finally:
        audit._directory_lock = original


@contextmanager
def idle_pool(plan, user):
    root = Path(plan["output_dir"])
    # These are the existing GPU controller's locks. Do not modify its state.
    with existing_source_lock(root), existing_source_lock(root / "pool"):
        registry = root / "submitted-jobs.txt"
        known = set(registry.read_text().splitlines()) if registry.exists() else set()
        state = root / "pool/controller.json"
        if state.exists():
            saved = read(state)
            require(
                not saved.get("submission_in_progress"),
                "GPU submission outcome uncertain",
            )
            known.update(str(saved.get(k, "")) for k in ("active_gpu", "audit_job"))
        active = pool.queued(user) & known
        require(not active, f"Scoring/audit jobs are still queued: {sorted(active)}")
        yield


def prepare(path, output, web, expected_parts):
    """Caller holds the pool and output locks; freeze before submitting Slurm."""
    plan = pool.single.load_plan(path)
    validate_output(output, path, plan, web)
    binding = {
        "version": "completed-youtube-snapshot-v1",
        "plan_path": str(path),
        "plan_sha256": _sha256(path),
        "output": str(output),
        "web": str(web),
        "expected_parts": expected_parts,
        "code": identity(),
        "scope_decision": "Researcher stopped remaining filtering on 2026-10-21; completed partitions only. No new GPU work.",
    }
    scope_path = output / "scope.json"
    if scope_path.exists():
        scope = read(scope_path)
        require(
            all(scope.get(k) == v for k, v in binding.items()),
            "Snapshot binding changed",
        )
        require(
            scope["frozen_sha256"] == _sha256(output / "audit/frozen.json"),
            "Frozen subset changed",
        )
        return scope
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    frozen = audit.freeze(path, plan, audit_dir)
    require(
        len(frozen["parts"]) == expected_parts,
        "Completed-part count differs from requested snapshot; inspect frozen.json",
    )
    slots = sorted(p["slot"] for p in frozen["parts"])
    scope = {
        **binding,
        "completed_slots": slots,
        "unprocessed_slots": sorted(set(audit.source_slots(plan)) - set(slots)),
        "frozen_sha256": _sha256(audit_dir / "frozen.json"),
    }
    _write_json(scope_path, scope)
    return scope


def audit_snapshot(path, output, scope, threads=8):
    require(
        scope["frozen_sha256"] == _sha256(output / "audit/frozen.json"),
        "Frozen subset changed",
    )
    checkpoint = output / "import-audit.json"
    if checkpoint.exists():
        saved = read(checkpoint)
        require(
            saved.get("complete") is True
            and saved["scope_sha256"] == _sha256(output / "scope.json")
            and saved["plan_sha256"] == _sha256(path)
            and saved["completed_slots"] == scope["completed_slots"]
            and saved["audit_summary_sha256"]
            == _sha256(output / "audit/review/summary.json"),
            "Snapshot audit checkpoint changed",
        )
        return read(output / "audit/review/summary.json")
    plan = pool.single.load_plan(path)
    with source_locks_without_creation(plan), parallel_fingerprints(threads):
        report = audit.execute(path, output / "audit")
    require(
        report["audit_complete"]
        and report["completed_parts"] == len(scope["completed_slots"])
        and report["unprocessed_slots"] == scope["unprocessed_slots"],
        "Audit did not cover exactly the frozen subset",
    )
    _write_json(
        checkpoint,
        {
            "complete": True,
            "plan_path": str(path),
            "plan_sha256": _sha256(path),
            "completed_slots": scope["completed_slots"],
            "unprocessed_slots": scope["unprocessed_slots"],
            "scope_sha256": _sha256(output / "scope.json"),
            "audit_summary_sha256": _sha256(output / "audit/review/summary.json"),
            "scope": scope["scope_decision"],
        },
    )
    return report


def submit(output, user, env):
    target = output / "submission.json"
    state = read(target) if target.exists() else {}
    require(
        not state.get("submission_in_progress"),
        "CPU submission outcome uncertain; inspect squeue before retrying",
    )
    if state.get("job_id") in pool.queued(user):
        LOG.info("Snapshot CPU job %s is already queued", state["job_id"])
        return state["job_id"]
    if (output / "checks-complete.json").exists():
        LOG.info(
            "CPU stages already completed; see %s", output / "checks-complete.json"
        )
        return None
    state.update(
        submission_in_progress=True, scope_sha256=_sha256(output / "scope.json")
    )
    _write_json(target, state)
    # Explicit CPU resources and a scrubbed Slurm option environment prevent an
    # inherited SBATCH_GRES/GPUS option from turning this into a GPU allocation.
    clean = {k: v for k, v in env.items() if not k.startswith("SBATCH_")}
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--time=04:00:00",
            "--cpus-per-task=8",
            "--mem=32G",
            f"--output={output}/youtube-snapshot-%j.log",
            str(HERE / "job.sbatch"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clean,
    )
    identifier = pool.job_id(result.stdout)
    state.update(submission_in_progress=False, job_id=identifier)
    _write_json(target, state)
    LOG.info(
        "Submitted CPU job %s: 8 CPUs, 32 GiB, no GPUs. Log: %s/youtube-snapshot-%s.log",
        identifier,
        output,
        identifier,
    )
    return identifier


def execute(args):
    path, output, web = (p.resolve() for p in (args.plan, args.output, args.web))
    plan = pool.single.load_plan(path)
    validate_output(output, path, plan, web)
    output.mkdir(parents=True, exist_ok=True)
    if args.status:
        for name in (
            "scope.json",
            "submission.json",
            "progress.json",
            "audit/progress.json",
            "summary.json",
            "checks-complete.json",
        ):
            if (output / name).exists():
                print(f"{output / name}\n{(output / name).read_text()}")
        return
    with idle_pool(plan, args.user):
        if not args.work:
            with _directory_lock(output):
                prepare(path, output, web, args.expected_parts)
                env = {
                    **os.environ,
                    "YT_SNAPSHOT_OUTPUT": str(output),
                    "SINGLE_PLAN": str(path),
                    "YT_SNAPSHOT_WEB": str(web),
                    "YT_SNAPSHOT_EXPECTED_PARTS": str(args.expected_parts),
                }
                submit(output, args.user, env)
            return
        scope = prepare(path, output, web, args.expected_parts)
        # Keep the controller locks across both subprocess stages, including the
        # environment switch to the existing dedup Python installation.
        audit_snapshot(path, output, scope)
        subprocess.run(
            [
                os.environ["YT_SNAPSHOT_DEDUP_PYTHON"],
                "-u",
                "-m",
                "scripts.youtube_cpu_io.run",
                "--threads",
                "8",
                "finalize",
                "--output",
                str(output),
                "--web",
                str(web),
                "--workers",
                "7",
            ],
            check=True,
        )
        subprocess.run(
            [
                os.environ["CURATION_CPU_PYTHON"],
                "-u",
                "-m",
                "scripts.release.record_rights",
                "--manifest",
                str(output / "corpus-manifest.json"),
                "--output",
                str(output / "rights"),
                "--prepared-config",
                str(Path(plan["input_dir"]) / "run-config.json"),
            ],
            check=True,
        )
        subprocess.run(
            [
                os.environ["CURATION_CPU_PYTHON"],
                "-u",
                "-m",
                "scripts.release.audit_snapshot",
                "--manifest",
                str(output / "corpus-manifest.json"),
                "--output",
                str(output / "integrity"),
                "--rights",
                str(output / "rights"),
                "--workers",
                "7",
            ],
            check=True,
        )
        _write_json(
            output / "checks-complete.json",
            {
                "cpu_stages_complete": True,
                "release_ready": False,
                "summary_sha256": _sha256(output / "summary.json"),
                "rights_summary_sha256": _sha256(output / "rights/summary.json"),
                "integrity_summary_sha256": _sha256(output / "integrity/summary.json"),
                "remaining": "Resolve rights evidence and notices, human quality review, and publication integrity gates before upload.",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--expected-parts", type=int, default=1834)
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--work", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.expected_parts > 0, "Expected completed parts must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args)


if __name__ == "__main__":
    main()
