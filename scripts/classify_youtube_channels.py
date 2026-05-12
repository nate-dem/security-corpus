"""Classify YouTube channels as security-relevant using a two-tier term system.

Tier 1 — CORE: terms derived from the normalized seed bank
(MITRE ATT&CK tactics/techniques, NIST CSRC glossary, Security NLP labels).
Overly generic seed bank entries (e.g. "software", "hardware", "dns") are
excluded via _GENERIC_EXCLUSIONS to keep precision high at channel-name level.

Tier 2 — ADJACENT: security-aligned domains endorsed by Dr. Karbasi
(alignment, privacy, cryptography, adversarial ML, surveillance, etc.) that
are not covered by the core MITRE/NIST taxonomies but are in scope for the
corpus.

Both tiers use whole-word, case-insensitive regex matching against channel names.

Output:
  data/youtube-transcripts/security_channels.txt   — channel_id per line (backward-compatible)
  data/youtube-transcripts/security_channels_detail.tsv — channel_id, name, tier, matched_term

Usage:
    python scripts/classify_youtube_channels.py
"""
import csv
import re
from pathlib import Path

CHANNEL_CSV = (
    Path("data/youtube-transcripts/all_channels.csv")
    if Path("data/youtube-transcripts/all_channels.csv").exists()
    else Path("data/youtube-transcripts/channels_shard0.csv")
)
SEED_BANK_CSV = Path("data/security_seed_bank_package/normalized_seed_bank_starter.csv")
KNOWN_CHANNELS_CSV = Path("data/security_seed_bank_package/youtube_security_channel_allowlist.csv")
OUT_TXT = Path("data/youtube-transcripts/security_channels.txt")
OUT_TSV = Path("data/youtube-transcripts/security_channels_detail.tsv")

# Entry types from the seed bank that are meaningful at the channel-name level.
_USEFUL_ENTRY_TYPES = {"tactic", "technique_or_subtechnique", "glossary_term", "label_schema"}

# Seed bank terms that are syntactically valid but far too generic to use
# against channel names without causing large numbers of false positives.
_GENERIC_EXCLUSIONS = frozenset({
    # ATT&CK tactics — single common words
    "execution", "persistence", "discovery", "collection", "impact", "stealth",
    # ATT&CK techniques — infrastructure/generic nouns
    "hardware", "software", "firmware", "server", "domains", "dns", "botnet",
    "credentials", "email addresses", "employee names", "ip addresses",
    "social media", "search engines", "code repositories", "network topology",
    "network trust dependencies", "business relationships", "identify roles",
    "determine physical locations", "identify business tempo",
    "tool", "artificial intelligence", "vulnerabilities",
    "generate content", "written content", "audio-visual content",
    "acquire infrastructure", "establish accounts",
    # NIST glossary — too short or too broad
    "2fa", "3des", "abac", "access point", "access point name",
    "acceptable risk", "acceptable use agreement", "access authority",
    "access complexity", "access control entry", "access control list",
    "access control matrix", "access control mechanism", "access control model",
    "access control policy", "access control",
    # Security NLP labels — too broad
    "action", "entity", "modifier", "system", "organization",
    "discover", "patch",
})

# Tier 2: security-adjacent domains in scope per Dr. Karbasi.
# These are not in the MITRE/NIST taxonomies but are explicitly in scope.
_ADJACENT_TERMS = frozenset({
    # AI safety / alignment — use phrases, not bare "alignment" (too ambiguous)
    "ai alignment", "ai safety", "ai security", "adversarial ml",
    "adversarial machine learning", "robustness", "interpretability",
    "trustworthy ai", "responsible ai",
    # Privacy engineering
    "privacy", "differential privacy", "anonymization", "data protection",
    "gdpr", "ccpa", "pii",
    # Cryptography (distinct from "crypto" which is dominated by cryptocurrency)
    "cryptography", "cryptographic", "encryption", "decryption",
    "public key", "pki", "tls", "ssl", "certificate authority",
    "hash function", "digital signature",
    # Surveillance / counter-surveillance
    "surveillance", "counter-surveillance", "wiretap", "eavesdropping",
    # Policy / governance
    "compliance", "nist", "iso 27001", "risk management", "cyber policy",
    # Core security terms not well-covered by ATT&CK names alone
    "security", "cyber", "infosec", "netsec",
    "hacking", "hacker", "ethical hacking",
    "pentest", "penetration testing", "penetration", "red team", "blue team", "purple team",
    "ctf", "capture the flag",
    "malware", "ransomware", "spyware", "rootkit", "trojan", "worm", "virus",
    "exploit", "exploitation", "vulnerability", "zero day", "zeroday", "0day",
    "cve", "cwe", "osint", "recon", "forensic", "dfir",
    "incident response", "threat", "threat hunting", "threat intelligence", "threat intel",
    "reverse engineering", "reverse engineer",
    "siem", "soar", "edr", "xdr",
    "firewall", "intrusion", "intrusion detection", "intrusion prevention", "ids", "ips",
    "bugbounty", "bug bounty",
    "defcon", "def con", "black hat", "blackhat", "sans", "bsides",
    "kali", "nmap", "metasploit", "burp suite", "wireshark",
})


def _load_known_channels() -> dict[str, str]:
    """Load hand-curated known security channels with confirmed channel IDs.

    Returns {channel_id: channel_name} for rows where youtube_channel_id is
    populated and needs_channel_id_resolution is 'no'.
    """
    if not KNOWN_CHANNELS_CSV.exists():
        return {}
    known = {}
    with open(KNOWN_CHANNELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("youtube_channel_id", "").strip()
            needs_resolution = row.get("needs_channel_id_resolution", "yes").strip().lower()
            if cid and needs_resolution == "no":
                known[cid] = row["channel_name"].strip()
    return known


def _load_seed_bank_terms() -> frozenset[str]:
    """Load core terms from the seed bank CSV, filtered to channel-name-safe entries."""
    if not SEED_BANK_CSV.exists():
        return frozenset()
    terms = set()
    with open(SEED_BANK_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["entry_type"] not in _USEFUL_ENTRY_TYPES:
                continue
            t = row["normalized_term"].strip().lower()
            if t and t not in _GENERIC_EXCLUSIONS:
                terms.add(t)
    return frozenset(terms)


def _build_regex(terms: frozenset[str]) -> re.Pattern:
    # Sort longest first so multi-word phrases match before their substrings.
    sorted_terms = sorted(terms, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def classify_channel(name: str, core_re: re.Pattern, adjacent_re: re.Pattern) -> tuple[str, str] | None:
    """Return (tier, matched_term) if the channel name matches, else None."""
    m = core_re.search(name)
    if m:
        return "core", m.group(0).lower()
    m = adjacent_re.search(name)
    if m:
        return "adjacent", m.group(0).lower()
    return None


def main() -> None:
    if not CHANNEL_CSV.exists():
        print(f"Channel CSV not found: {CHANNEL_CSV}")
        print("Run scripts/extract_youtube_channels.py first.")
        return

    known_channels = _load_known_channels()
    core_terms = _load_seed_bank_terms()
    if not core_terms:
        print(f"Warning: seed bank not found at {SEED_BANK_CSV} — core tier will be empty.")

    core_re = _build_regex(core_terms)
    adjacent_re = _build_regex(_ADJACENT_TERMS)

    print(f"Known channels (confirmed IDs)   : {len(known_channels):,}")
    print(f"Core terms loaded from seed bank : {len(core_terms):,}")
    print(f"Adjacent terms                   : {len(_ADJACENT_TERMS):,}")

    total = 0
    results: list[tuple[str, str, str, str]] = []  # (channel_id, name, tier, matched_term)
    seen_ids: set[str] = set()

    # Known channels go in unconditionally — add them first.
    for cid, name in sorted(known_channels.items()):
        results.append((cid, name, "known", "hand-curated"))
        seen_ids.add(cid)

    with open(CHANNEL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            cid = row["channel_id"]
            if cid in seen_ids:
                continue  # already included via known list
            match = classify_channel(row["channel_name"], core_re, adjacent_re)
            if match:
                tier, term = match
                results.append((cid, row["channel_name"], tier, term))
                seen_ids.add(cid)

    matched = len(results)
    known_count = sum(1 for r in results if r[2] == "known")
    core_count = sum(1 for r in results if r[2] == "core")
    adjacent_count = sum(1 for r in results if r[2] == "adjacent")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    # Backward-compatible flat list for the connector.
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Security-relevant YouTube channel IDs\n")
        f.write(f"# Classified from: {CHANNEL_CSV.name} + known allowlist\n")
        f.write(f"# Channels: {matched} from channel CSV ({total:,} scanned) + {known_count} known\n")
        f.write(f"# Known: {known_count}  |  Core (seed bank): {core_count}  |  Adjacent: {adjacent_count}\n")
        for cid, _, _, _ in sorted(results, key=lambda r: r[0]):
            f.write(f"{cid}\n")

    # Auditable detail file with tier and matched term.
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["channel_id", "channel_name", "tier", "matched_term"])
        for row in sorted(results, key=lambda r: r[0]):
            writer.writerow(row)

    print(f"\nClassified {total:,} channels — {matched:,} total security-relevant")
    print(f"  Known (hand-curated) : {known_count}")
    print(f"  Core (seed bank)     : {core_count}")
    print(f"  Adjacent             : {adjacent_count}")
    print(f"\nWritten to {OUT_TXT}")
    print(f"Written to {OUT_TSV}")
    print("\nSample matched channels:")
    for cid, name, tier, term in results[:25]:
        print(f"  [{tier:8s}] {term:30s}  {name}")


if __name__ == "__main__":
    main()
