"""Default source configuration for Security Scope."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INDEX_DIR = Path("data/security-paperclip-index/v1")
INDEX_ENV_VAR = "SECURITYCLIP_INDEX"


def default_index_dir() -> Path:
    """Return the configured index path, honoring SECURITYCLIP_INDEX."""

    value = os.environ.get(INDEX_ENV_VAR)
    if value:
        return Path(value).expanduser()
    return DEFAULT_INDEX_DIR


@dataclass(frozen=True)
class SourceSpec:
    """A Parquet input pattern and the root it should appear under."""

    name: str
    pattern: str


FINAL_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("arxiv-cs-cr-full", "data/final-no-chunk/academic_papers/arxiv_cs_cr_full.parquet"),
    SourceSpec("arxiv-citation-qwen-kept-full", "data/final-no-chunk/academic_papers/arxiv_citation_qwen_kept_full.parquet"),
    SourceSpec("capec", "data/training-clean-v2/normalized/source_id=capec/*.parquet"),
    SourceSpec("cisa-kev", "data/training-clean-v2/normalized/source_id=cisa-kev/*.parquet"),
    SourceSpec("mitre-attack", "data/training-clean-v2/normalized/source_id=mitre-attack/*.parquet"),
    SourceSpec("mitre-cwe", "data/training-clean-v2/normalized/source_id=mitre-cwe/*.parquet"),
    SourceSpec("nvd", "data/training-clean-v2/normalized/source_id=nvd/*.parquet"),
    SourceSpec("sigma", "data/final-no-chunk/normalized/source_id=sigma/*.parquet"),
    SourceSpec("cloudtrail-flaws", "data/final-no-chunk/normalized/source_id=cloudtrail-flaws/*.parquet"),
    SourceSpec("youtube-transcripts", "data/training-clean-v2/normalized/source_id=youtube-transcripts/*.parquet"),
    SourceSpec("stackoverflow", "data/filtering/v3/qwen_qa_kept/source_id=stackoverflow/*.parquet"),
    SourceSpec("stackexchange-crypto", "data/filtering/v3/qwen_qa_kept/source_id=stackexchange-crypto/*.parquet"),
    SourceSpec("stackexchange-infosec", "data/filtering/v3/qwen_qa_kept/source_id=stackexchange-infosec/*.parquet"),
    SourceSpec(
        "stackexchange-reverseengineering",
        "data/filtering/v3/qwen_qa_kept/source_id=stackexchange-reverseengineering/*.parquet",
    ),
    SourceSpec("stackexchange-tor", "data/filtering/v3/qwen_qa_kept/source_id=stackexchange-tor/*.parquet"),
    SourceSpec("reddit-antivirus", "data/filtering/v3/qwen_qa_kept/source_id=reddit-antivirus/*.parquet"),
    SourceSpec("reddit-asknetsec", "data/filtering/v3/qwen_qa_kept/source_id=reddit-asknetsec/*.parquet"),
    SourceSpec("reddit-blueteamsec", "data/filtering/v3/qwen_qa_kept/source_id=reddit-blueteamsec/*.parquet"),
    SourceSpec("reddit-bugbounty", "data/filtering/v3/qwen_qa_kept/source_id=reddit-bugbounty/*.parquet"),
    SourceSpec("reddit-cloudflare", "data/filtering/v3/qwen_qa_kept/source_id=reddit-cloudflare/*.parquet"),
    SourceSpec("reddit-computersecurity", "data/filtering/v3/qwen_qa_kept/source_id=reddit-computersecurity/*.parquet"),
    SourceSpec("reddit-computerviruses", "data/filtering/v3/qwen_qa_kept/source_id=reddit-computerviruses/*.parquet"),
    SourceSpec("reddit-cryptography", "data/filtering/v3/qwen_qa_kept/source_id=reddit-cryptography/*.parquet"),
    SourceSpec("reddit-cyberlaws", "data/filtering/v3/qwen_qa_kept/source_id=reddit-cyberlaws/*.parquet"),
    SourceSpec("reddit-cybersecurity", "data/filtering/v3/qwen_qa_kept/source_id=reddit-cybersecurity/*.parquet"),
    SourceSpec(
        "reddit-cybersecurity_help",
        "data/filtering/v3/qwen_qa_kept/source_id=reddit-cybersecurity_help/*.parquet",
    ),
    SourceSpec(
        "reddit-cybersecurityadvice",
        "data/filtering/v3/qwen_qa_kept/source_id=reddit-cybersecurityadvice/*.parquet",
    ),
    SourceSpec("reddit-hacking", "data/filtering/v3/qwen_qa_kept/source_id=reddit-hacking/*.parquet"),
    SourceSpec("reddit-malware", "data/filtering/v3/qwen_qa_kept/source_id=reddit-malware/*.parquet"),
    SourceSpec("reddit-netsec", "data/filtering/v3/qwen_qa_kept/source_id=reddit-netsec/*.parquet"),
    SourceSpec("reddit-netsecstudents", "data/filtering/v3/qwen_qa_kept/source_id=reddit-netsecstudents/*.parquet"),
    SourceSpec("reddit-phishing", "data/filtering/v3/qwen_qa_kept/source_id=reddit-phishing/*.parquet"),
    SourceSpec(
        "reddit-reverseengineering",
        "data/filtering/v3/qwen_qa_kept/source_id=reddit-reverseengineering/*.parquet",
    ),
    SourceSpec("reddit-security", "data/filtering/v3/qwen_qa_kept/source_id=reddit-security/*.parquet"),
    SourceSpec("reddit-tor", "data/filtering/v3/qwen_qa_kept/source_id=reddit-tor/*.parquet"),
    SourceSpec("reddit-vpn", "data/filtering/v3/qwen_qa_kept/source_id=reddit-vpn/*.parquet"),
)

FINAL_DATA_STRUCTURED_SOURCES: tuple[str, ...] = (
    "capec",
    "cisa-kev",
    "cloudtrail-flaws",
    "mitre-attack",
    "mitre-cwe",
    "nvd",
    "sigma",
    "youtube-transcripts",
    "fineweb-security",
)

FINAL_DATA_QA_SOURCES: tuple[str, ...] = (
    "stackoverflow",
    "stackexchange-crypto",
    "stackexchange-infosec",
    "stackexchange-reverseengineering",
    "stackexchange-tor",
    "reddit-antivirus",
    "reddit-asknetsec",
    "reddit-blueteamsec",
    "reddit-bugbounty",
    "reddit-cloudflare",
    "reddit-computersecurity",
    "reddit-computerviruses",
    "reddit-cryptography",
    "reddit-cyberlaws",
    "reddit-cybersecurity",
    "reddit-cybersecurity_help",
    "reddit-cybersecurityadvice",
    "reddit-hacking",
    "reddit-malware",
    "reddit-netsec",
    "reddit-netsecstudents",
    "reddit-phishing",
    "reddit-reverseengineering",
    "reddit-security",
    "reddit-tor",
    "reddit-vpn",
)

FINAL_DATA_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("arxiv-cs-cr-full", "final-data/academic_papers/arxiv_cs_cr_full.parquet"),
    SourceSpec("arxiv-citation-qwen-kept-full", "final-data/academic_papers/arxiv_citation_qwen_kept_full.parquet"),
    *(SourceSpec(source_id, f"final-data/normalized/source_id={source_id}/*.parquet") for source_id in FINAL_DATA_STRUCTURED_SOURCES),
    *(SourceSpec(source_id, f"final-data/qwen_qa_kept/source_id={source_id}/*.parquet") for source_id in FINAL_DATA_QA_SOURCES),
)

SOURCE_LAYOUTS: dict[str, tuple[SourceSpec, ...]] = {
    "repo": FINAL_SOURCE_SPECS,
    "final-data": FINAL_DATA_SOURCE_SPECS,
}
