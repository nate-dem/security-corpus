"""YouTube video cybersecurity classifier.

This module handles everything related to the vLLM-based metadata classifier:

  - ClassificationResult  : the structured output schema for one classified video
  - SYSTEM_PROMPT         : the constant instruction block sent to the model
  - _build_user_message() : formats one record's metadata into the user turn
  - _parse_response()     : JSON parsing with a three-level fallback chain
  - _LENIENT_FALLBACK     : the default result used when all parsing fails (keep the video)
  - load_keep_set()       : reads the merged cache; returns the set of video_ids to keep
  - load_full_cache()     : reads the merged cache; returns full ClassificationResult per video_id

The classifier is designed to run on Marlowe HPC (Slurm + vLLM) via
scripts/classify_youtube_videos.py.  This module itself contains no vLLM
imports so it is safe to import on any machine (including macOS dev boxes).

Cache file format
-----------------
One JSON object per line.  Each line produced by classify_youtube_videos.py
looks like:

    {
        "video_id": "abc123",
        "channel_id": "UCxxx",
        "channel": "SecurityAcademy",
        "title": "SQL Injection Deep Dive",
        "word_count": 4200,
        "result": {
            "is_cybersecurity_relevant": true,
            "confidence": 0.91,
            "relevance_level": "high",
            "reason": "Deep-dive SQL injection walkthrough",
            "topic_tags": ["sql injection", "web security"],
            "should_keep": true
        },
        "classified_at": "2026-05-20T14:00:00Z",
        "model": "Qwen/Qwen3-8B"
    }

load_keep_set() reads this file and returns just the set of video_ids whose
result.should_keep is true.  iter_records() in youtube_transcripts.py checks
against this set at gate time — O(1) per record, no ClassificationResult
objects kept in memory during ingest.
"""

import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Structured output for one classified YouTube video.

    Fields
    ------
    is_cybersecurity_relevant
        True if the video touches cybersecurity at all, including borderline topics.
    confidence
        Model's self-reported confidence, clamped to [0.0, 1.0].
    relevance_level
        Coarse bucket.  "high" = clearly security; "medium" = security-adjacent;
        "low" = weak signal but plausible; "not_relevant" = clearly off-topic.
    reason
        Short explanation (≤ 30 words) of why the model made this call.
    topic_tags
        Up to 5 short tags the model assigned (e.g. "sql injection", "red team").
    should_keep
        The single gate bit consumed by iter_records().  True unless
        relevance_level is "not_relevant" AND confidence ≥ 0.70.
        If confidence < 0.50 the model must also set this to true regardless of
        relevance_level — the prompt enforces this.
    """
    is_cybersecurity_relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_level: Literal["high", "medium", "low", "not_relevant"]
    reason: str
    topic_tags: list[str] = Field(default_factory=list)
    should_keep: bool

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Guard against models that return 1.1 or -0.05."""
        return max(0.0, min(1.0, v))


# Used whenever a response cannot be parsed after all retries.
# Keeps the video — we would rather include noise than drop valid content.
_LENIENT_FALLBACK = ClassificationResult(
    is_cybersecurity_relevant=True,
    confidence=0.30,
    relevance_level="low",
    reason="parse failure — keeping by default",
    topic_tags=[],
    should_keep=True,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The system message is constant across all records in a batch.  vLLM's prefix
# caching reuses the KV representation of this block for every record in the
# shard, which cuts prefill cost by ~80%.
SYSTEM_PROMPT = """\
You are a cybersecurity content classifier for a research training corpus. \
Your sole task is to classify YouTube videos as cybersecurity-relevant or not, \
using only the lightweight metadata provided (title, channel name, word count, \
language, URL).

Be LENIENT. This corpus has a second-stage quality filter. False positives \
(keeping a non-security video) are acceptable. False negatives (dropping a \
valid security video) are the primary risk to avoid.

CLASSIFY AS RELEVANT if the video covers any of the following:
- Vulnerability research: CVE/CWE analysis, patch analysis, exploit development or mitigation
- Malware analysis: reverse engineering, binary analysis, sandbox reports, rootkits, ransomware, spyware
- Threat intelligence: CTI, threat hunting, threat modeling, MITRE ATT&CK, STIX/TAXII
- Network security: packet capture, firewall, IDS/IPS, VPN hardening, DNS security, DDoS defense
- Web application security: SQLi, XSS, CSRF, SSRF, auth bypass, API security, OWASP Top 10
- Cloud security: AWS/GCP/Azure hardening, Kubernetes security, container security, IAM, CSPM, CWPP
- Identity and access: authentication, authorization, OAuth 2.0, SSO, Zero Trust, MFA, PAM, RBAC
- Cryptography in security contexts: TLS/SSL, PKI, key management, AES, RSA, hashing, digital signatures
- Secure software engineering: SAST, DAST, threat modeling, secure code review, SDL, supply chain security
- Incident response and forensics: DFIR, memory forensics, log analysis, disk imaging, chain of custody
- Offensive security: penetration testing, red team operations, CTF, bug bounty, exploit development
- Defensive security: SOC operations, SIEM, EDR, SOAR, detection engineering, Sigma rules, blue team
- Security tooling: Burp Suite, Wireshark, Metasploit, Nmap, Ghidra, IDA Pro, Kali Linux, OSINT frameworks
- Privacy engineering: differential privacy, anonymization, PII handling, GDPR compliance from a security angle
- Governance and compliance: NIST CSF/SP 800-53, ISO 27001, SOC 2, FedRAMP, CIS Benchmarks, risk management
- Hardware and firmware security: embedded systems, IoT security, side-channel attacks, UEFI/BIOS hardening
- Security education: DEF CON / Black Hat / BSides / SANS talks, OSCP / CEH / Security+ / CISSP prep, \
CTF walkthroughs, lab tutorials

LEAN TOWARD KEEPING in borderline cases:
- General programming that explicitly covers: input validation, authentication flows, secrets management, \
permissions, SQL injection prevention, secure API design, dependency vulnerability scanning, or insecure deserialization
- DevOps/infrastructure content covering: secrets management, least privilege, network segmentation, \
hardening guides, or patch automation
- Cryptography tutorials (AES, RSA, elliptic curves, hashing) even without explicit "security" framing
- Any channel whose name contains: security, cyber, hack, exploit, CTF, pentest, OSCP, malware, \
forensic, reverse, infosec, red team, blue team, SOC, vulnerability

CLASSIFY AS NOT RELEVANT only when clearly off-topic:
- Pure programming tutorials with no security angle (build a CRUD app, learn Python syntax, React hooks)
- General networking without attack or defense context (home router setup, Wi-Fi troubleshooting)
- Cryptocurrency or blockchain content unless it covers smart contract auditing or cryptographic protocol design
- General data science or ML without adversarial ML, model security, or privacy engineering
- Gaming, entertainment, cooking, personal vlogs, fitness, lifestyle

UNCERTAINTY RULE: If the title is ambiguous, the channel name is generic, or the topic is unclear, \
default to keeping the video. Set should_keep to true and confidence between 0.30 and 0.49.

Respond with exactly one JSON object. No markdown code fences, no preamble, no text after the closing brace:
{"is_cybersecurity_relevant": <bool>, "confidence": <float 0.0-1.0>, \
"relevance_level": <"high"|"medium"|"low"|"not_relevant">, "reason": "<30 words or fewer>", \
"topic_tags": [<up to 5 short strings>], "should_keep": <bool>}

Constraints:
- should_keep must be true when relevance_level is "high", "medium", or "low"
- should_keep must be false only when relevance_level is "not_relevant" AND confidence >= 0.70
- If confidence < 0.50, set should_keep to true regardless of relevance_level\
"""


def _build_user_message(record: dict, description: str | None = None) -> str:
    """Format one record's metadata into the user turn sent to the model.

    Only lightweight metadata is included — no transcript text.  The model
    receives what a human skimming a YouTube search result would see.

    description, when provided, comes from Rijgersberg/YouTube-Commons-descriptions
    (a HuggingFace dataset built from the same PleIAs/YouTube-Commons corpus).
    It is truncated to 300 characters to avoid inflating per-record token count
    while still capturing the most informative part of the description.
    """
    title = record.get("title") or "(no title)"
    channel = record.get("channel") or "(unknown channel)"
    word_count = record.get("word_count") or 0
    language = (
        record.get("transcription_language")
        or record.get("original_language")
        or "unknown"
    )
    url = record.get("video_link") or ""

    lines = [
        f"Title: {title}",
        f"Channel: {channel}",
        f"Word count: {word_count}",
        f"Language: {language}",
        f"URL: {url}",
    ]
    if description:
        lines.append(f"Description: {description[:300]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> ClassificationResult | None:
    """Try to extract a ClassificationResult from a raw model response string.

    Three attempts in order:
      1. Parse the entire response as JSON directly.
      2. Regex-extract the first {...} block and parse that.
      3. Return None — the caller falls back to _LENIENT_FALLBACK.

    The function never raises; it always returns either a valid
    ClassificationResult or None.
    """
    # Strip qwen3 thinking blocks (<think>...</think>) before parsing.
    # These appear when enable_thinking is not fully suppressed.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Attempt 1: the model returned only JSON (ideal case).
    try:
        return ClassificationResult(**json.loads(cleaned))
    except Exception:
        pass

    # Attempt 2: the model wrapped the JSON in prose or code fences.
    # Grab the first {...} block (greedy to handle nested braces in reason strings).
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return ClassificationResult(**json.loads(match.group()))
        except Exception:
            pass

    # All attempts failed.
    return None


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_keep_set(cache_path: Path) -> frozenset[str]:
    """Read the merged classifier cache and return the set of video_ids to keep.

    This is the function called by iter_records() at ingest time.  It reads the
    JSONL file once, collects all video_ids where result.should_keep is true,
    and returns a frozenset for O(1) membership tests at gate time.

    If the cache file does not exist, returns an empty frozenset.  The caller
    (iter_records) treats an empty keep set as "no classifier cache available"
    and passes all records that meet the language + word count gates.

    Note: load_keep_set returns an empty set (drop everything not classified)
    only if the file exists but is empty.  If the file is absent entirely, the
    caller should pass video_keep_set=None, not an empty frozenset.  See
    iter_records() for how it handles both cases.
    """
    if not cache_path.exists():
        logger.warning("Classifier cache not found at %s — gate will pass all records", cache_path)
        return frozenset()

    keep: set[str] = set()
    with open(cache_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("result", {}).get("should_keep", True):
                    vid = entry.get("video_id")
                    if vid:
                        keep.add(vid)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed cache line %d", lineno)

    logger.info("Loaded %d kept video_ids from %s", len(keep), cache_path)
    return frozenset(keep)


def load_full_cache(cache_path: Path) -> dict[str, ClassificationResult]:
    """Read the merged classifier cache and return a full ClassificationResult per video_id.

    Used by merge_classifier_caches.py and audit scripts that need the full
    result (confidence, tags, reason) rather than just the keep bit.  For
    ingest-time gate checks, prefer load_keep_set() which is cheaper.
    """
    if not cache_path.exists():
        return {}

    cache: dict[str, ClassificationResult] = {}
    with open(cache_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                vid = entry.get("video_id")
                result_dict = entry.get("result")
                if vid and result_dict:
                    cache[vid] = ClassificationResult(**result_dict)
            except Exception:
                logger.debug("Skipping malformed cache line %d", lineno)

    return cache
