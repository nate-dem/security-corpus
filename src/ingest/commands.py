"""Command-line entrypoints for corpus ingestion."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ingest.connectors.reddit import REDDIT_SUBREDDITS
from ingest.connectors.stackexchange.stackoverflow import StackOverflowConnector
from ingest.pipeline import ingest, ingest_and_store
from ingest.writers import write_parquet


@dataclass(frozen=True)
class SimpleSourcePlan:
    source: str
    default_input: Path
    default_output: Path
    label: str
    aliases: tuple[str, ...] = ()
    missing_hint: str | None = None


MITRE_ATTACK_DOMAINS = (
    "enterprise-attack",
    "mobile-attack",
    "ics-attack",
    "pre-attack",
)

STACKEXCHANGE_SITES = {
    "infosec": "stackexchange-infosec",
    "reverseengineering": "stackexchange-reverseengineering",
    "crypto": "stackexchange-crypto",
    "tor": "stackexchange-tor",
}

SIMPLE_SOURCES = (
    SimpleSourcePlan(
        source="cisa-kev",
        default_input=Path("data/cisa-kev/raw/known_exploited_vulnerabilities.json"),
        default_output=Path("data/cisa-kev/normalized"),
        label="CISA KEV",
    ),
    SimpleSourcePlan(
        source="mitre-cwe",
        default_input=Path("data/mitre-cwe/raw/cwec_v4.19.1.xml"),
        default_output=Path("data/mitre-cwe/normalized"),
        label="MITRE CWE",
    ),
    SimpleSourcePlan(
        source="capec",
        default_input=Path("data/mitre-capec/raw/capec_latest.xml"),
        default_output=Path("data/mitre-capec/normalized"),
        label="CAPEC",
        aliases=("mitre-capec",),
    ),
    SimpleSourcePlan(
        source="sigma",
        default_input=Path("data/sigma/raw"),
        default_output=Path("data/sigma/normalized"),
        label="sigma",
    ),
    SimpleSourcePlan(
        source="bron",
        default_input=Path("data/bron/raw"),
        default_output=Path("data/bron/normalized"),
        label="bron",
        missing_hint="Run python scripts/export/bron.py first.",
    ),
    SimpleSourcePlan(
        source="github-advisory",
        default_input=Path("data/github-advisory/raw"),
        default_output=Path("data/github-advisory/normalized"),
        label="github-advisory",
        missing_hint="Run python scripts/export/github_advisory.py first.",
    ),
    SimpleSourcePlan(
        source="youtube-transcripts",
        default_input=Path("data/youtube-transcripts/raw"),
        default_output=Path("data/youtube-transcripts/normalized"),
        label="youtube-transcripts",
        missing_hint="Run python scripts/export/youtube_transcripts.py first.",
    ),
    SimpleSourcePlan(
        source="arxiv",
        default_input=Path("data/arxiv/raw"),
        default_output=Path("data/arxiv/normalized"),
        label="arxiv",
        missing_hint=(
            "Run the arXiv preprocessing scripts first: "
            "scripts/arxiv/harvest_metadata.py, "
            "scripts/arxiv/download_sources.py, "
            "scripts/arxiv/normalize_sources.py."
        ),
    ),
)


def _path_arg(value: str) -> Path:
    return Path(value)


def _ingest_simple(args: argparse.Namespace) -> int:
    plan: SimpleSourcePlan = args.plan
    input_path = args.input or plan.default_input
    output_dir = args.output_dir or plan.default_output

    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        if plan.missing_hint:
            print(plan.missing_hint)
        return 1

    count = ingest_and_store(input_path, source=plan.source, output_dir=output_dir)
    print(f"{plan.label}: {count} records")
    return 0


def _ingest_nvd(args: argparse.Namespace) -> int:
    raw_dir = args.raw_dir
    output_dir = args.output_dir

    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        return 1

    files = sorted(raw_dir.glob("nvdcve-2.0-[0-9]*.json.gz"))
    if not files:
        print(f"No NVD year files found in {raw_dir}")
        return 1

    total = 0
    for path in files:
        count = ingest_and_store(path, source="nvd", output_dir=output_dir)
        print(f"{path.name}: {count} records")
        total += count

    print(f"\nTotal: {total} records across {len(files)} files")
    return 0


def _ingest_mitre_attack(args: argparse.Namespace) -> int:
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    domains = args.domains or MITRE_ATTACK_DOMAINS

    total = 0
    seen = 0
    for domain in domains:
        bundle = raw_dir / domain / f"{domain}.json"
        if not bundle.exists():
            print(f"Skipping {domain}: {bundle} not found")
            continue
        count = ingest_and_store(bundle, source="mitre-attack", output_dir=output_dir)
        print(f"{domain}: {count} records")
        total += count
        seen += 1

    if seen == 0:
        print(f"No MITRE ATT&CK bundles found in {raw_dir}")
        return 1

    print(f"\nTotal: {total} records across {seen} domains")
    return 0


def _ingest_stackexchange(args: argparse.Namespace) -> int:
    source_id = STACKEXCHANGE_SITES[args.site]
    raw_dir = args.raw_dir or Path(f"data/{source_id}/raw/extracted")
    output_dir = args.output_dir or Path(f"data/{source_id}/normalized")

    if not raw_dir.is_dir():
        print(f"Extracted directory not found: {raw_dir}")
        return 1

    count = ingest_and_store(raw_dir, source=source_id, output_dir=output_dir)
    print(f"{source_id}: {count} records")
    return 0


def _ingest_stackoverflow(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if shutil.which("7z") is None:
        print("Error: 7z not found. Install with: brew install p7zip")
        return 1

    if not args.archive_path.exists():
        print(f"Archive not found: {args.archive_path}")
        print("Make sure the flash drive is mounted at /Volumes/SECURITY/")
        return 1

    connector = StackOverflowConnector()
    count = connector.ingest(
        archive_path=args.archive_path,
        output_dir=args.output_dir,
        intermediate_dir=args.intermediate_dir,
    )
    print(f"stackoverflow: {count} records")
    return 0


def _ingest_reddit(args: argparse.Namespace) -> int:
    if args.all:
        subreddits = REDDIT_SUBREDDITS
    elif args.subreddit:
        subreddits = [args.subreddit]
    else:
        raise SystemExit("Provide a subreddit name or --all")

    wrote_any = False
    for subreddit in subreddits:
        source_id = f"reddit-{subreddit.lower()}"
        submissions_file = args.data_dir / f"{subreddit}_submissions.zst"
        if not submissions_file.exists():
            print(f"Skipping {subreddit}: {submissions_file} not found")
            continue

        records = ingest(args.data_dir, source=source_id)
        count = write_parquet(
            records,
            args.output_dir,
            source=source_id,
            input_path=Path(subreddit.lower()),
        )
        print(f"{source_id}: {count} records")
        wrote_any = True

    return 0 if wrote_any else 1


def _ingest_cloudtrail(args: argparse.Namespace) -> int:
    if not args.data_dir.is_dir():
        print(f"CloudTrail data directory not found: {args.data_dir}")
        return 1

    source_id = "cloudtrail-flaws"
    records = ingest(args.data_dir, source=source_id)
    count = write_parquet(
        records,
        args.output_dir,
        source=source_id,
        input_path=Path("flaws"),
    )
    print(f"{source_id}: {count} records")
    return 0


def _list_sources(_: argparse.Namespace) -> int:
    print("Ingestion commands:")
    print("  nvd")
    print("  mitre-attack")
    for plan in SIMPLE_SOURCES:
        names = ", ".join((plan.source, *plan.aliases))
        print(f"  {names}")
    print("  stackexchange {infosec,reverseengineering,crypto,tor}")
    print("  stackoverflow")
    print("  reddit {subreddit|--all}")
    print("  cloudtrail-flaws")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest security corpus sources into normalized Parquet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available sources.")
    list_parser.set_defaults(handler=_list_sources)

    nvd_parser = subparsers.add_parser("nvd", help="Ingest NVD yearly JSON feeds.")
    nvd_parser.add_argument("--raw-dir", type=_path_arg, default=Path("data/nvd/raw"))
    nvd_parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/nvd/normalized"),
    )
    nvd_parser.set_defaults(handler=_ingest_nvd)

    attack_parser = subparsers.add_parser(
        "mitre-attack",
        help="Ingest MITRE ATT&CK domain bundles.",
    )
    attack_parser.add_argument(
        "--raw-dir",
        type=_path_arg,
        default=Path("data/mitre-attack/raw"),
    )
    attack_parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/mitre-attack/normalized"),
    )
    attack_parser.add_argument(
        "--domains",
        nargs="+",
        choices=MITRE_ATTACK_DOMAINS,
        help="Specific ATT&CK domains to ingest.",
    )
    attack_parser.set_defaults(handler=_ingest_mitre_attack)

    for plan in SIMPLE_SOURCES:
        source_parser = subparsers.add_parser(
            plan.source,
            aliases=list(plan.aliases),
            help=f"Ingest {plan.label}.",
        )
        source_parser.add_argument(
            "--input",
            type=_path_arg,
            help=f"Input file or directory (default: {plan.default_input}).",
        )
        source_parser.add_argument(
            "--output-dir",
            type=_path_arg,
            help=f"Output directory (default: {plan.default_output}).",
        )
        source_parser.set_defaults(handler=_ingest_simple, plan=plan)

    se_parser = subparsers.add_parser(
        "stackexchange",
        help="Ingest one small Stack Exchange site dump.",
    )
    se_parser.add_argument("site", choices=sorted(STACKEXCHANGE_SITES))
    se_parser.add_argument("--raw-dir", type=_path_arg)
    se_parser.add_argument("--output-dir", type=_path_arg)
    se_parser.set_defaults(handler=_ingest_stackexchange)

    so_parser = subparsers.add_parser(
        "stackoverflow",
        help="Stream-ingest Stack Overflow from the .7z archive.",
    )
    so_parser.add_argument(
        "--archive-path",
        type=_path_arg,
        default=Path("/Volumes/SECURITY/stackoverflow/raw/stackoverflow.com.7z"),
    )
    so_parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/stackoverflow/normalized"),
    )
    so_parser.add_argument(
        "--intermediate-dir",
        type=_path_arg,
        default=Path("/Volumes/SECURITY/stackoverflow/intermediate"),
    )
    so_parser.set_defaults(handler=_ingest_stackoverflow)

    reddit_parser = subparsers.add_parser(
        "reddit",
        help="Ingest Reddit subreddit data from Arctic Shift dumps.",
    )
    reddit_parser.add_argument(
        "subreddit",
        nargs="?",
        help="Subreddit name to ingest (case-sensitive, matches filename).",
    )
    reddit_parser.add_argument("--all", action="store_true")
    reddit_parser.add_argument(
        "--data-dir",
        type=_path_arg,
        default=Path("data/reddit/raw"),
    )
    reddit_parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/reddit/normalized"),
    )
    reddit_parser.set_defaults(handler=_ingest_reddit)

    cloudtrail_parser = subparsers.add_parser(
        "cloudtrail-flaws",
        aliases=["cloudtrail"],
        help="Ingest flaws.cloud CloudTrail logs.",
    )
    cloudtrail_parser.add_argument(
        "--data-dir",
        type=_path_arg,
        default=Path("data/cloudtrail/raw/flaws_cloudtrail_logs"),
    )
    cloudtrail_parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=Path("data/cloudtrail/normalized"),
    )
    cloudtrail_parser.set_defaults(handler=_ingest_cloudtrail)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

