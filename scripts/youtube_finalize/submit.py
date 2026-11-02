"""Fetch existing bundles on the login node, then submit one resumable CPU job."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def call(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", action="store_true", help="Read local progress; submit nothing"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    curation = Path(os.environ.get("CYBEROPEN_WORK_DIR", project / "data")) / "curation"
    defaults = {
        "YT_FINISH_PROJECT": project,
        "YT_FINISH_OUTPUT": curation / "youtube-finalized-v1",
        "YT_FINISH_MANIFEST": curation / "modal-youtube-v1/manifest.json",
        "YT_FINISH_WEB": curation / "web-near-dedup-v1",
        "YT_FINISH_CLIENT": project / "scripts/modal_youtube/.venv/bin/python",
        "YT_FINISH_AUDIT_PYTHON": project / "scripts/web_corpora/.venv/bin/python",
        "YT_FINISH_PYTHON": project / "scripts/web_semantic/.venv/bin/python",
    }
    env = {**os.environ, **{k: os.environ.get(k, str(v)) for k, v in defaults.items()}}
    env["PYTHONPATH"] = f"{project / 'src'}:{project}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    output = Path(env["YT_FINISH_OUTPUT"]).resolve()
    if args.status:
        for name in (
            "submitted-jobs.txt",
            "progress.json",
            "audit/progress.json",
            "dedup/progress.json",
            "summary.json",
        ):
            path = output / name
            if path.exists():
                print(f"{path}\n{path.read_text()}")
        return
    for path in (
        project,
        Path(env["YT_FINISH_MANIFEST"]).parent,
        Path(env["YT_FINISH_WEB"]),
    ):
        path = path.resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(
                "Finalization output must be separate from existing inputs"
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = call(
            ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"], capture_output=True
        ).stdout
        active = {line.split("_")[0].strip() for line in queue.splitlines()}
        jobs = output / "submitted-jobs.txt"
        if jobs.exists() and active.intersection(jobs.read_text().splitlines()):
            print("A finalization job is already active; no duplicate submitted.")
            return
        uncertain = output / "submission-in-progress"
        if uncertain.exists():
            raise RuntimeError(
                f"Earlier sbatch outcome is uncertain; inspect squeue and {uncertain}"
            )
        if (output / "summary.json").exists():
            report = json.loads((output / "summary.json").read_text())
            if report.get("complete"):
                print(json.dumps(report, indent=2))
                print("Recorded finalization is complete; no new job submitted.")
                return
        call(
            [
                env["YT_FINISH_AUDIT_PYTHON"],
                "-c",
                "from scripts.youtube_finalize import pipeline; from scripts.curation_audit import run; "
                "from scripts.modal_youtube import transfer",
            ],
            cwd=project,
            env=env,
        )
        call(
            [
                env["YT_FINISH_PYTHON"],
                "-c",
                "from scripts.youtube_finalize import finalize; finalize.dedup.runtime()",
            ],
            cwd=project,
            env=env,
        )
        manifest = Path(env["YT_FINISH_MANIFEST"])
        if not (manifest.parent / "returned/downloaded.json").exists():
            print(
                "Fetching already-built recovery bundles; no new Modal compute.",
                flush=True,
            )
            call(
                [
                    env["YT_FINISH_CLIENT"],
                    project / "scripts/modal_youtube_recover.py",
                    "fetch",
                    "--manifest",
                    manifest,
                ],
                cwd=project,
                env=env,
            )
        else:
            print(
                "Existing download found; CPU import will verify it before use.",
                flush=True,
            )
        # Verify a full recovery, not the normal stopped-run download marker.
        call(
            [
                env["YT_FINISH_CLIENT"],
                "-c",
                "from pathlib import Path; import sys; "
                "from scripts.youtube_finalize.pipeline import recovered_slots; "
                "print('Recovered partitions:', len(recovered_slots(Path(sys.argv[1]))))",
                manifest,
            ],
            cwd=project,
            env=env,
        )
        (project / "logs/youtube-finalize").mkdir(parents=True, exist_ok=True)
        uncertain.write_text(
            "Submission attempted; check squeue before removing this marker.\n"
        )
        # Keep the marker on ambiguous errors to prevent duplicate allocations.
        reply = call(
            [
                "sbatch",
                "--parsable",
                "--time",
                env.get("YT_FINISH_TIME", "04:00:00"),
                project / "scripts/youtube_finalize/job.sbatch",
            ],
            cwd=project,
            env=env,
            capture_output=True,
        ).stdout.strip()
        job = reply.split(";")[0]
        if not re.fullmatch(r"[0-9]+", job):
            raise RuntimeError(f"Unexpected sbatch reply {reply!r}; inspect squeue")
        with jobs.open("a") as handle:
            handle.write(job + "\n")
        uncertain.unlink()
        print(f"Submitted CPU job {job}: 8 CPUs, 32 GB, no GPUs.")
        print(f"Log: {project}/logs/youtube-finalize/youtube-finalize-{job}.log")
        print(f"Final count: {output}/summary.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
