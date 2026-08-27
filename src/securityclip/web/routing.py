"""Deterministic query routing before optional model routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from securityclip.paths import normalize_virtual_path


CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
VIRTUAL_PATH_RE = re.compile(r"(?P<path>/(?:papers|qa|vulns|rules|logs|knowledge|transcripts|web)(?:/[A-Za-z0-9_.:-]+)*/?)")

ROOT_ALIASES: dict[str, str] = {
    "paper": "/papers",
    "papers": "/papers",
    "arxiv": "/papers",
    "qa": "/qa",
    "q&a": "/qa",
    "stackoverflow": "/qa",
    "stack overflow": "/qa",
    "reddit": "/qa",
    "vuln": "/vulns",
    "vulns": "/vulns",
    "vulnerability": "/vulns",
    "vulnerabilities": "/vulns",
    "nvd": "/vulns",
    "kev": "/vulns",
    "cisa": "/vulns",
    "rule": "/rules",
    "rules": "/rules",
    "sigma": "/rules",
    "log": "/logs",
    "logs": "/logs",
    "cloudtrail": "/logs",
    "knowledge": "/knowledge",
    "mitre": "/knowledge",
    "attack": "/knowledge",
    "cwe": "/knowledge",
    "capec": "/knowledge",
    "transcript": "/transcripts",
    "transcripts": "/transcripts",
    "youtube": "/transcripts",
    "web": "/web",
    "fineweb": "/web",
    "fineweb-security": "/web",
    "web corpus": "/web",
}

KNOWN_SOURCE_IDS = {
    "arxiv",
    "nvd",
    "cisa-kev",
    "mitre-attack",
    "mitre-cwe",
    "capec",
    "sigma",
    "cloudtrail-flaws",
    "youtube-transcripts",
    "fineweb-security",
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
}


@dataclass(frozen=True)
class Route:
    route_type: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    likely_roots: list[str] = field(default_factory=list)
    confidence: float = 0.0
    deterministic: bool = False
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "route_type": self.route_type,
            "entities": self.entities,
            "likely_roots": self.likely_roots,
            "confidence": self.confidence,
            "deterministic": self.deterministic,
            "reason": self.reason,
        }


def deterministic_route(query: str) -> Route | None:
    normalized = " ".join(query.strip().split())
    lowered = normalized.lower()
    if not normalized:
        return Route("general", deterministic=True, confidence=1.0, reason="empty query")

    paths = [normalize_virtual_path(match.group("path")) for match in VIRTUAL_PATH_RE.finditer(normalized)]
    if paths:
        return Route(
            "path_inspection",
            entities={"paths": paths},
            likely_roots=sorted({_root_for_path(path) for path in paths}),
            confidence=0.99,
            deterministic=True,
            reason="query contains virtual path",
        )

    cves = sorted({match.group(0).upper() for match in CVE_RE.finditer(normalized)})
    if cves:
        route_type = "source_list" if any(term in lowered for term in ("source", "sources", "include", "including", "list")) else "exact_identifier"
        return Route(
            route_type,
            entities={"cve_ids": cves},
            likely_roots=["/vulns", "/qa", "/rules", "/papers", "/knowledge"],
            confidence=0.98,
            deterministic=True,
            reason="query contains CVE identifier",
        )

    arxiv_ids = sorted({match.group(0) for match in ARXIV_RE.finditer(normalized)})
    if arxiv_ids:
        return Route(
            "exact_identifier",
            entities={"arxiv_ids": arxiv_ids},
            likely_roots=["/papers"],
            confidence=0.98,
            deterministic=True,
            reason="query contains arXiv identifier",
        )

    roots = sorted({root for token, root in ROOT_ALIASES.items() if re.search(rf"\b{re.escape(token)}\b", lowered)})
    sources = sorted({source for source in KNOWN_SOURCE_IDS if source in lowered})
    if roots or sources:
        route_type = "paper_research" if roots == ["/papers"] else "log_or_rule_search" if any(root in roots for root in ("/logs", "/rules")) else "topic_research"
        return Route(
            route_type,
            entities={"source_ids": sources} if sources else {},
            likely_roots=roots,
            confidence=0.8,
            deterministic=True,
            reason="query contains source/root mention",
        )

    return None


def deterministic_operations(query: str, route: Route, *, max_results: int) -> list[dict]:
    if route.route_type == "path_inspection":
        operations: list[dict] = []
        for path in route.entities.get("paths", [])[:3]:
            if path.endswith("/content.lines"):
                operations.append({"tool": "head", "path": path, "count": 80})
            elif path.endswith("/meta.json") or path.endswith("/rule.yml"):
                operations.append({"tool": "cat", "path": path})
            else:
                operations.append({"tool": "ls", "path": path, "limit": max_results})
        return operations

    cves = route.entities.get("cve_ids", [])
    if cves:
        return [{"tool": "search", "query": cve, "limit": max_results} for cve in cves[:2]]

    arxiv_ids = route.entities.get("arxiv_ids", [])
    if arxiv_ids:
        return [{"tool": "search", "query": arxiv_id, "limit": max_results} for arxiv_id in arxiv_ids[:2]]

    if route.likely_roots:
        root = route.likely_roots[0]
        return [{"tool": "search", "query": query, "limit": max_results}, {"tool": "grep", "pattern": _short_literal(query), "path": root + "/", "ignore_case": True, "limit": max_results}]

    return []


def _short_literal(query: str) -> str:
    cleaned = re.sub(r"\b(find|all|documents|related|to|give|me|a|list|of|sources|that|include|includes)\b", " ", query, flags=re.IGNORECASE)
    return " ".join(cleaned.split()) or query


def _root_for_path(path: str) -> str:
    parts = normalize_virtual_path(path).split("/")
    return "/" + parts[1] if len(parts) > 1 and parts[1] else "/"
