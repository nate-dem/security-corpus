"""Classify YouTube channels as security-relevant using keyword matching on channel names.

Reads data/youtube-transcripts/channels_shard0.csv (or all_channels.csv when available),
classifies each channel by name, and writes matching channel IDs to
data/youtube-transcripts/security_channels.txt.

This is the first-pass heuristic classifier. When running at scale on S3,
replace or augment with a model-based classifier over the full 721k channel list.

Usage:
    python scripts/classify_youtube_channels.py
"""
import csv
import re
from pathlib import Path

# Use the full channel list if available, otherwise fall back to shard 0 sample.
CHANNEL_CSV = (
    Path("data/youtube-transcripts/all_channels.csv")
    if Path("data/youtube-transcripts/all_channels.csv").exists()
    else Path("data/youtube-transcripts/channels_shard0.csv")
)
OUT = Path("data/youtube-transcripts/security_channels.txt")

# Keywords matched against channel name (whole-word, case-insensitive).
SECURITY_CHANNEL_KEYWORDS = frozenset({
    "security", "cyber", "hacking", "hacker", "hack",
    "infosec", "pentest", "penetration", "ctf", "malware",
    "forensic", "exploit", "vulnerability", "osint", "recon",
    "red team", "blue team", "threat", "incident response",
    "reverse engineering", "reverse engineer", "dfir",
    "cryptography", "crypto",  # channel-name crypto is usually technical
    "privacy", "surveillance",
    "defcon", "def con", "black hat", "blackhat", "sans",
    "bugbounty", "bug bounty", "kali", "nmap", "metasploit",
    "firewall", "intrusion", "siem", "netsec",
    "zero day", "zeroday", "0day", "cve", "cwe",
    "ethical", "capture the flag",
})

_RE = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in SECURITY_CHANNEL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_security_channel(name: str) -> bool:
    return bool(_RE.search(name))


def main() -> None:
    if not CHANNEL_CSV.exists():
        print(f"Channel CSV not found: {CHANNEL_CSV}")
        print("Run scripts/extract_youtube_channels.py first.")
        return

    total = matched = 0
    results: list[tuple[str, str]] = []

    with open(CHANNEL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if is_security_channel(row["channel_name"]):
                results.append((row["channel_id"], row["channel_name"]))
                matched += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Security-relevant YouTube channel IDs\n")
        f.write(f"# Classified from: {CHANNEL_CSV.name}\n")
        f.write(f"# Channels: {matched} of {total} ({matched/total*100:.1f}%)\n")
        for cid, _ in sorted(results):
            f.write(f"{cid}\n")

    print(f"Classified {total:,} channels — {matched:,} security-relevant ({matched/total*100:.1f}%)")
    print(f"Written to {OUT}")
    print()
    print("Sample matched channels:")
    for cid, name in results[:25]:
        print(f"  {cid}  {name}")


if __name__ == "__main__":
    main()
