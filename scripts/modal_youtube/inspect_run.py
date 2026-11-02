"""Export small scheduling/timing metadata before a possible Modal migration.

Standard library only. Reads worker summaries, never transcript contents. Does
not submit jobs, upload data, install packages, or spend Modal credits. These
checks do not replace the source/raw-response integrity audit.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def inspect(plan_path):
    raw = plan_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    root = Path(plan["output_dir"])
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest,
        "model": plan["model"],
        "settings": plan["settings"],
        "completed_parts": 0,
        "completed_input_tokens": 0,
        "candidate_tokens": 0,
        "unresolved_documents": 0,
        "remaining_parts": 0,
        "remaining_input_tokens": 0,
        "completed_last_invocation_wall_seconds": 0.0,
        "completed_last_invocation_generation_seconds": 0.0,
        "completed_last_invocation_attempted_spans": 0,
        "parts": [],
        "scope": "Scheduling metadata, not an integrity audit or transfer authorization. Remaining tokens include whole unfinished parts even if some responses are cached. Invocation timings exclude earlier attempts, lost work, setup before partitions and idle time; they are not total GPU billing. Workers may change during this read; freeze a fresh allocation before migration.",
    }
    for slot, part in enumerate(plan["parts"]):
        if Path(part["path"]).parts[1] != "youtube-commons":
            continue
        summary_path = root / "parts" / f"{slot:05d}" / "summary.json"
        summary_raw = summary_path.read_bytes() if summary_path.exists() else None
        state = json.loads(summary_raw) if summary_raw else {}
        if state and state.get("plan_sha256") != digest:
            raise ValueError(f"Wrong plan binding in slot {slot}")
        finished = state.get("workflow_finished") is True
        if finished:
            if (
                state.get("source") != "youtube-commons"
                or state.get("slot") != slot
                or state.get("input_tokens") != part["tokens"]
                or state.get("records") != part["records"]
            ):
                raise ValueError(f"Worker counts/identity differ in slot {slot}")
            result["completed_parts"] += 1
            result["completed_input_tokens"] += state["input_tokens"]
            result["candidate_tokens"] += state["candidate_tokens"]
            result["unresolved_documents"] += state["unresolved_documents"]
            wall = state["wall_seconds_this_invocation"]
            generations = state["generation_this_invocation"]
            seconds = sum(g["generation_seconds"] for g in generations)
            if any(not math.isfinite(v) or v < 0 for v in (wall, seconds)):
                raise ValueError(f"Invalid timing in slot {slot}")
            result["completed_last_invocation_wall_seconds"] += wall
            result["completed_last_invocation_generation_seconds"] += seconds
            result["completed_last_invocation_attempted_spans"] += sum(
                g["attempted_spans"] for g in generations
            )
        else:
            result["remaining_parts"] += 1
            result["remaining_input_tokens"] += part["tokens"]
        result["parts"].append(
            {
                "slot": slot,
                "path": part["path"],
                "input_sha256": part["sha256"],
                "input_tokens": part["tokens"],
                "workflow_finished": finished,
                "summary_sha256": hashlib.sha256(summary_raw).hexdigest()
                if summary_raw
                else None,
            }
        )
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != digest:
        raise ValueError("Plan changed while reading summaries")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.plan)
    # Do not overwrite an existing report, input, or checkpoint by mistake.
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, indent=2))


if __name__ == "__main__":
    main()
