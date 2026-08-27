"""Build a linked Tier 1 vulnerability/knowledge graph dataset.

This sidecar dataset connects the current Tier 1 sources without changing the
normalized ingestion schema:

    NVD/CISA KEV CVE -> CWE -> CAPEC -> MITRE ATT&CK technique -> Sigma rule

The output is meant for multi-hop reasoning experiments. It writes four
Parquet files:

* nodes.parquet: source-record nodes from NVD, KEV, CWE, CAPEC, ATT&CK, Sigma
* edges.parquet: deterministic relationships extracted from normalized fields
  and preserved raw MITRE XML/JSON
* chains.parquet: compact CVE-to-technique paths for direct querying/sampling
* detection_chains.parquet: expanded CVE-to-Sigma paths where rules exist
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from lxml import etree
import pyarrow as pa
import pyarrow.parquet as pq


_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_ATTACK_ID_RE = re.compile(r"^T?\d{4}(?:\.\d{3})?$", re.IGNORECASE)
_SIGMA_ATTACK_TAG_RE = re.compile(r"\battack\.t(\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_XML_PARSER = etree.XMLParser(recover=True, resolve_entities=False)

_CLEAN_V2_NORMALIZED_SUFFIX = ("training-clean-v2", "normalized")
_CLEAN_V2_REASONING_OUTPUT_DIR_NAME = "tier1-reasoning-clean-v2"

_SOURCE_GLOBS = {
    "nvd": "source_id=nvd/*.parquet",
    "cisa-kev": "source_id=cisa-kev/*.parquet",
    "mitre-cwe": "source_id=mitre-cwe/*.parquet",
    "capec": "source_id=capec/*.parquet",
    "mitre-attack": "source_id=mitre-attack/*.parquet",
    "sigma": "source_id=sigma/*.parquet",
}

_COMMON_COLUMNS = [
    "source_id",
    "source_record_id",
    "record_id",
    "content",
    "title",
    "content_length",
    "source_url",
    "license",
    "raw",
]

_VULN_COLUMNS = [
    *_COMMON_COLUMNS,
    "cve_id",
    "severity",
    "cvss_score",
    "cwe_ids",
    "exploited_in_wild",
]

_SIGMA_COLUMNS = [
    *_COMMON_COLUMNS,
    "rule_id",
    "rule_level",
    "rule_source",
]

_DETECTION_CHAIN_SCHEMA = pa.schema(
    [
        ("detection_chain_id", pa.string()),
        ("base_chain_id", pa.string()),
        ("cve_id", pa.string()),
        ("nvd_node_id", pa.string()),
        ("cisa_kev_node_id", pa.string()),
        ("is_known_exploited", pa.bool_()),
        ("cwe_id", pa.string()),
        ("cwe_title", pa.string()),
        ("capec_id", pa.string()),
        ("capec_title", pa.string()),
        ("attack_technique_id", pa.string()),
        ("attack_technique_title", pa.string()),
        ("sigma_rule_id", pa.string()),
        ("sigma_rule_title", pa.string()),
        ("sigma_rule_level", pa.string()),
        ("sigma_node_id", pa.string()),
        ("sigma_source_url", pa.string()),
        ("path_node_ids", pa.list_(pa.string())),
        ("path_relationships", pa.list_(pa.string())),
        ("evidence_edge_ids", pa.list_(pa.string())),
        ("chain_text", pa.string()),
    ]
)


@dataclass(frozen=True)
class Tier1BuildResult:
    """Summary of a Tier 1 link dataset build."""

    output_dir: Path
    nodes: int
    edges: int
    chains: int
    detection_chains: int
    nodes_by_type: dict[str, int]
    edges_by_relationship: dict[str, int]
    complete_kev_chains: int
    chains_with_sigma: int
    complete_kev_detection_chains: int


def build_tier1_reasoning_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    max_chains: int | None = None,
) -> Tier1BuildResult:
    """Build Tier 1 nodes, edges, and reasoning chains from normalized Parquet."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _validate_clean_v2_reasoning_output_dir(output_dir)
    input_files = _resolve_input_files(data_dir)

    nodes: dict[str, dict[str, Any]] = {}
    nvd_by_cve: dict[str, str] = {}
    kev_by_cve: dict[str, str] = {}
    cve_to_nodes: dict[str, set[str]] = defaultdict(set)
    vulnerability_cwe_edges: list[tuple[str, str, str, str]] = []

    _load_vulnerability_nodes(
        input_files["nvd"],
        nodes=nodes,
        cve_to_nodes=cve_to_nodes,
        source_cve_index=nvd_by_cve,
        vulnerability_cwe_edges=vulnerability_cwe_edges,
    )
    _load_vulnerability_nodes(
        input_files["cisa-kev"],
        nodes=nodes,
        cve_to_nodes=cve_to_nodes,
        source_cve_index=kev_by_cve,
        vulnerability_cwe_edges=vulnerability_cwe_edges,
    )

    cwe_relations = _load_cwe_nodes(input_files["mitre-cwe"], nodes=nodes)
    capec_relations = _load_capec_nodes(input_files["capec"], nodes=nodes)
    attack_ids = _load_attack_technique_nodes(input_files["mitre-attack"], nodes=nodes)
    sigma_attack_index = _load_sigma_rule_nodes(input_files["sigma"], nodes=nodes)

    edges = _build_edges(
        nodes=nodes,
        nvd_by_cve=nvd_by_cve,
        kev_by_cve=kev_by_cve,
        cve_to_nodes=cve_to_nodes,
        vulnerability_cwe_edges=vulnerability_cwe_edges,
        cwe_relations=cwe_relations,
        capec_relations=capec_relations,
        attack_ids=attack_ids,
        sigma_attack_index=sigma_attack_index,
    )
    chains = _build_chains(
        nodes=nodes,
        edges=edges,
        cve_to_nodes=cve_to_nodes,
        nvd_by_cve=nvd_by_cve,
        kev_by_cve=kev_by_cve,
        max_chains=max_chains,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(output_dir / "nodes.parquet", nodes.values())
    _write_parquet(output_dir / "edges.parquet", edges)
    _write_parquet(output_dir / "chains.parquet", chains)
    detection_chain_stats = _write_detection_chains(
        output_dir / "detection_chains.parquet",
        nodes=nodes,
        edges=edges,
        chains=chains,
    )

    nodes_by_type = Counter(node["entity_type"] for node in nodes.values())
    edges_by_relationship = Counter(edge["relationship_type"] for edge in edges)
    complete_kev_chains = sum(1 for chain in chains if chain["cisa_kev_node_id"])
    chains_with_sigma = sum(1 for chain in chains if chain["sigma_rule_count"])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {source: [str(path) for path in paths] for source, paths in input_files.items()},
        "outputs": {
            "nodes": str(output_dir / "nodes.parquet"),
            "edges": str(output_dir / "edges.parquet"),
            "chains": str(output_dir / "chains.parquet"),
            "detection_chains": str(output_dir / "detection_chains.parquet"),
        },
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "chains": len(chains),
            "detection_chains": detection_chain_stats["rows"],
            "complete_kev_chains": complete_kev_chains,
            "chains_with_sigma": chains_with_sigma,
            "complete_kev_detection_chains": detection_chain_stats["complete_kev_rows"],
            "nodes_by_type": dict(sorted(nodes_by_type.items())),
            "edges_by_relationship": dict(sorted(edges_by_relationship.items())),
        },
        "relationship_path": [
            "nvd/cisa-kev CVE",
            "mitre-cwe CWE",
            "capec attack pattern",
            "mitre-attack technique",
            "sigma detection rule",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return Tier1BuildResult(
        output_dir=output_dir,
        nodes=len(nodes),
        edges=len(edges),
        chains=len(chains),
        detection_chains=detection_chain_stats["rows"],
        nodes_by_type=dict(sorted(nodes_by_type.items())),
        edges_by_relationship=dict(sorted(edges_by_relationship.items())),
        complete_kev_chains=complete_kev_chains,
        chains_with_sigma=chains_with_sigma,
        complete_kev_detection_chains=detection_chain_stats["complete_kev_rows"],
    )


def _resolve_input_files(data_dir: Path) -> dict[str, list[Path]]:
    _validate_clean_v2_normalized_dir(data_dir)
    input_files: dict[str, list[Path]] = {}
    missing: list[str] = []
    for source, pattern in _SOURCE_GLOBS.items():
        files = sorted(data_dir.glob(pattern))
        if not files:
            missing.append(f"{source}: {data_dir / pattern}")
        input_files[source] = files
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing Tier 1 normalized Parquet inputs:\n  {joined}")
    return input_files


def _validate_clean_v2_normalized_dir(data_dir: Path) -> None:
    resolved_parts = data_dir.resolve().parts
    if tuple(resolved_parts[-2:]) != _CLEAN_V2_NORMALIZED_SUFFIX:
        expected = Path(*_CLEAN_V2_NORMALIZED_SUFFIX)
        raise ValueError(
            "Tier 1 reasoning input must be the clean-v2 normalized dataset "
            f"(path ending in {expected}), got {data_dir}."
        )


def _validate_clean_v2_reasoning_output_dir(output_dir: Path) -> None:
    if output_dir.name != _CLEAN_V2_REASONING_OUTPUT_DIR_NAME:
        raise ValueError(
            "Tier 1 reasoning output must be the clean-v2 output directory "
            f"named {_CLEAN_V2_REASONING_OUTPUT_DIR_NAME}, got {output_dir}."
        )


def _load_vulnerability_nodes(
    paths: Iterable[Path],
    *,
    nodes: dict[str, dict[str, Any]],
    cve_to_nodes: dict[str, set[str]],
    source_cve_index: dict[str, str],
    vulnerability_cwe_edges: list[tuple[str, str, str, str]],
) -> None:
    for row in _iter_parquet_rows(paths, _VULN_COLUMNS):
        cve_id = _normalize_cve_id(row.get("cve_id") or row.get("source_record_id"))
        if not cve_id:
            continue

        record_id = row["record_id"]
        source_id = row["source_id"]
        node = {
            "node_id": record_id,
            "entity_type": "cve",
            "entity_id": cve_id,
            "source_id": source_id,
            "source_record_id": row.get("source_record_id"),
            "record_id": record_id,
            "title": row.get("title"),
            "content": row.get("content"),
            "content_length": row.get("content_length"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "severity": row.get("severity"),
            "cvss_score": row.get("cvss_score"),
            "exploited_in_wild": row.get("exploited_in_wild"),
            "attack_domains": [],
            "attack_tactics": [],
            "rule_level": None,
            "sigma_attack_tags": [],
        }
        nodes[record_id] = node
        source_cve_index[cve_id] = record_id
        cve_to_nodes[cve_id].add(record_id)

        for cwe_id in _normalize_id_list(row.get("cwe_ids"), _normalize_cwe_id):
            vulnerability_cwe_edges.append((record_id, cwe_id, source_id, row.get("record_id") or record_id))


def _load_cwe_nodes(paths: Iterable[Path], *, nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    relations: dict[str, dict[str, Any]] = {}
    for row in _iter_parquet_rows(paths, _COMMON_COLUMNS + ["category_id"]):
        cwe_id = _normalize_cwe_id(row.get("category_id") or row.get("source_record_id"))
        if not cwe_id:
            continue

        record_id = row["record_id"]
        raw = _decode_raw(row.get("raw"))
        raw_xml = raw.get("raw_xml") if isinstance(raw, dict) else None
        parsed = _extract_cwe_xml_relations(raw_xml or "")
        relations[cwe_id] = parsed

        nodes[record_id] = {
            "node_id": record_id,
            "entity_type": "cwe",
            "entity_id": cwe_id,
            "source_id": row["source_id"],
            "source_record_id": row.get("source_record_id"),
            "record_id": record_id,
            "title": row.get("title"),
            "content": row.get("content"),
            "content_length": row.get("content_length"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "severity": None,
            "cvss_score": None,
            "exploited_in_wild": None,
            "attack_domains": [],
            "attack_tactics": [],
            "rule_level": None,
            "sigma_attack_tags": [],
        }
    return relations


def _load_capec_nodes(paths: Iterable[Path], *, nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    relations: dict[str, dict[str, Any]] = {}
    for row in _iter_parquet_rows(paths, _COMMON_COLUMNS + ["category_id"]):
        capec_id = _normalize_capec_id(row.get("category_id") or row.get("source_record_id"))
        if not capec_id:
            continue

        record_id = row["record_id"]
        raw = _decode_raw(row.get("raw"))
        raw_xml = raw.get("raw_xml") if isinstance(raw, dict) else None
        parsed = _extract_capec_xml_relations(raw_xml or "")
        relations[capec_id] = parsed

        nodes[record_id] = {
            "node_id": record_id,
            "entity_type": "capec",
            "entity_id": capec_id,
            "source_id": row["source_id"],
            "source_record_id": row.get("source_record_id"),
            "record_id": record_id,
            "title": row.get("title"),
            "content": row.get("content"),
            "content_length": row.get("content_length"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "severity": None,
            "cvss_score": None,
            "exploited_in_wild": None,
            "attack_domains": [],
            "attack_tactics": [],
            "rule_level": None,
            "sigma_attack_tags": [],
        }
    return relations


def _load_attack_technique_nodes(paths: Iterable[Path], *, nodes: dict[str, dict[str, Any]]) -> set[str]:
    attack_ids: set[str] = set()
    for row in _iter_parquet_rows(paths, _COMMON_COLUMNS + ["category_id"]):
        raw = _decode_raw(row.get("raw"))
        if raw.get("type") != "attack-pattern":
            continue

        attack_id = _normalize_attack_id(row.get("category_id") or row.get("source_record_id"))
        if not attack_id:
            continue

        record_id = f"mitre-attack:{attack_id}"
        attack_ids.add(attack_id)
        domains = sorted(_as_string_set(raw.get("x_mitre_domains")))
        tactics = sorted(
            phase.get("phase_name", "")
            for phase in raw.get("kill_chain_phases", [])
            if isinstance(phase, dict) and phase.get("phase_name")
        )

        existing = nodes.get(record_id)
        if existing:
            existing["attack_domains"] = sorted(set(existing["attack_domains"]) | set(domains))
            existing["attack_tactics"] = sorted(set(existing["attack_tactics"]) | set(tactics))
            continue

        nodes[record_id] = {
            "node_id": record_id,
            "entity_type": "attack-technique",
            "entity_id": attack_id,
            "source_id": row["source_id"],
            "source_record_id": attack_id,
            "record_id": record_id,
            "title": row.get("title"),
            "content": row.get("content"),
            "content_length": row.get("content_length"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "severity": None,
            "cvss_score": None,
            "exploited_in_wild": None,
            "attack_domains": domains,
            "attack_tactics": tactics,
            "rule_level": None,
            "sigma_attack_tags": [],
        }
    return attack_ids


def _load_sigma_rule_nodes(paths: Iterable[Path], *, nodes: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    sigma_attack_index: dict[str, set[str]] = defaultdict(set)
    for row in _iter_parquet_rows(paths, _SIGMA_COLUMNS):
        rule_id = str(row.get("rule_id") or row.get("source_record_id") or "").strip()
        if not rule_id:
            continue

        record_id = f"sigma:{rule_id}"
        attack_ids = _extract_sigma_attack_ids(row.get("rule_source") or row.get("content") or "")
        attack_tags = [f"attack.{attack_id.lower()}" for attack_id in attack_ids]
        nodes[record_id] = {
            "node_id": record_id,
            "entity_type": "sigma-rule",
            "entity_id": rule_id,
            "source_id": row["source_id"],
            "source_record_id": rule_id,
            "record_id": record_id,
            "title": row.get("title"),
            "content": row.get("content"),
            "content_length": row.get("content_length"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "severity": None,
            "cvss_score": None,
            "exploited_in_wild": None,
            "attack_domains": [],
            "attack_tactics": [],
            "rule_level": row.get("rule_level"),
            "sigma_attack_tags": attack_tags,
        }
        for attack_id in attack_ids:
            sigma_attack_index[attack_id].add(record_id)
    return sigma_attack_index


def _build_edges(
    *,
    nodes: dict[str, dict[str, Any]],
    nvd_by_cve: dict[str, str],
    kev_by_cve: dict[str, str],
    cve_to_nodes: dict[str, set[str]],
    vulnerability_cwe_edges: list[tuple[str, str, str, str]],
    cwe_relations: dict[str, dict[str, Any]],
    capec_relations: dict[str, dict[str, Any]],
    attack_ids: set[str],
    sigma_attack_index: dict[str, set[str]],
) -> list[dict[str, Any]]:
    edge_map: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}

    def add_edge(
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        *,
        evidence_source_id: str,
        evidence_record_id: str,
        evidence_field: str,
        evidence_detail: str | None = None,
    ) -> None:
        if source_node_id not in nodes or target_node_id not in nodes:
            return
        key = (
            source_node_id,
            target_node_id,
            relationship_type,
            evidence_source_id,
            evidence_record_id,
            evidence_field,
        )
        if key in edge_map:
            return
        source_node = nodes[source_node_id]
        target_node = nodes[target_node_id]
        edge_map[key] = {
            "edge_id": _stable_id("tier1-edge", *key),
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "relationship_type": relationship_type,
            "source_entity_type": source_node["entity_type"],
            "source_entity_id": source_node["entity_id"],
            "target_entity_type": target_node["entity_type"],
            "target_entity_id": target_node["entity_id"],
            "evidence_source_id": evidence_source_id,
            "evidence_record_id": evidence_record_id,
            "evidence_field": evidence_field,
            "evidence_detail": evidence_detail,
        }

    for cve_id, nvd_node_id in sorted(nvd_by_cve.items()):
        kev_node_id = kev_by_cve.get(cve_id)
        if not kev_node_id:
            continue
        add_edge(
            nvd_node_id,
            kev_node_id,
            "same_cve",
            evidence_source_id="cisa-kev",
            evidence_record_id=kev_node_id,
            evidence_field="cve_id",
            evidence_detail=cve_id,
        )
        add_edge(
            kev_node_id,
            nvd_node_id,
            "same_cve",
            evidence_source_id="nvd",
            evidence_record_id=nvd_node_id,
            evidence_field="cve_id",
            evidence_detail=cve_id,
        )

    for vulnerability_node_id, cwe_id, evidence_source_id, evidence_record_id in vulnerability_cwe_edges:
        add_edge(
            vulnerability_node_id,
            f"mitre-cwe:{cwe_id}",
            "has_weakness",
            evidence_source_id=evidence_source_id,
            evidence_record_id=evidence_record_id,
            evidence_field="cwe_ids",
            evidence_detail=cwe_id,
        )

    for cwe_id, relation in sorted(cwe_relations.items()):
        cwe_node_id = f"mitre-cwe:{cwe_id}"
        for related in relation["related_cwes"]:
            add_edge(
                cwe_node_id,
                f"mitre-cwe:{related['cwe_id']}",
                "related_weakness",
                evidence_source_id="mitre-cwe",
                evidence_record_id=cwe_node_id,
                evidence_field="raw_xml/Related_Weaknesses",
                evidence_detail=related.get("nature"),
            )
        for capec_id in relation["related_capecs"]:
            add_edge(
                cwe_node_id,
                f"capec:{capec_id}",
                "related_attack_pattern",
                evidence_source_id="mitre-cwe",
                evidence_record_id=cwe_node_id,
                evidence_field="raw_xml/Related_Attack_Patterns",
                evidence_detail=capec_id,
            )
        for cve_id in relation["observed_cves"]:
            for cve_node_id in cve_to_nodes.get(cve_id, ()):
                add_edge(
                    cve_node_id,
                    cwe_node_id,
                    "observed_example_of",
                    evidence_source_id="mitre-cwe",
                    evidence_record_id=cwe_node_id,
                    evidence_field="raw_xml/Observed_Examples",
                    evidence_detail=cve_id,
                )

    for capec_id, relation in sorted(capec_relations.items()):
        capec_node_id = f"capec:{capec_id}"
        for cwe_id in relation["related_cwes"]:
            add_edge(
                f"mitre-cwe:{cwe_id}",
                capec_node_id,
                "related_weakness_attack_pattern",
                evidence_source_id="capec",
                evidence_record_id=capec_node_id,
                evidence_field="raw_xml/Related_Weaknesses",
                evidence_detail=cwe_id,
            )
        for related in relation["related_capecs"]:
            add_edge(
                capec_node_id,
                f"capec:{related['capec_id']}",
                "related_attack_pattern",
                evidence_source_id="capec",
                evidence_record_id=capec_node_id,
                evidence_field="raw_xml/Related_Attack_Patterns",
                evidence_detail=related.get("nature"),
            )
        for attack_id in relation["attack_technique_ids"]:
            if attack_id in attack_ids:
                add_edge(
                    capec_node_id,
                    f"mitre-attack:{attack_id}",
                    "maps_to_attack_technique",
                    evidence_source_id="capec",
                    evidence_record_id=capec_node_id,
                    evidence_field="raw_xml/Taxonomy_Mappings/ATTACK",
                    evidence_detail=attack_id,
                )
        for cve_id in relation["example_cves"]:
            for cve_node_id in cve_to_nodes.get(cve_id, ()):
                add_edge(
                    cve_node_id,
                    capec_node_id,
                    "example_instance_of",
                    evidence_source_id="capec",
                    evidence_record_id=capec_node_id,
                    evidence_field="raw_xml/Example_Instances",
                    evidence_detail=cve_id,
                )

    for attack_id in sorted(attack_ids):
        if "." not in attack_id:
            continue
        parent_id = attack_id.split(".", 1)[0]
        if parent_id not in attack_ids:
            continue
        add_edge(
            f"mitre-attack:{attack_id}",
            f"mitre-attack:{parent_id}",
            "subtechnique_of",
            evidence_source_id="mitre-attack",
            evidence_record_id=f"mitre-attack:{attack_id}",
            evidence_field="category_id",
            evidence_detail=parent_id,
        )

    for attack_id, sigma_node_ids in sorted(sigma_attack_index.items()):
        if attack_id not in attack_ids:
            continue
        for sigma_node_id in sorted(sigma_node_ids):
            add_edge(
                f"mitre-attack:{attack_id}",
                sigma_node_id,
                "detected_by_sigma_rule",
                evidence_source_id="sigma",
                evidence_record_id=sigma_node_id,
                evidence_field="rule_source/tags",
                evidence_detail=f"attack.{attack_id.lower()}",
            )

    return sorted(edge_map.values(), key=lambda edge: edge["edge_id"])


def _build_chains(
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    cve_to_nodes: dict[str, set[str]],
    nvd_by_cve: dict[str, str],
    kev_by_cve: dict[str, str],
    max_chains: int | None,
) -> list[dict[str, Any]]:
    vuln_to_cwes: dict[str, set[str]] = defaultdict(set)
    cwe_to_capecs: dict[str, set[str]] = defaultdict(set)
    capec_to_attacks: dict[str, set[str]] = defaultdict(set)
    attack_to_sigmas: dict[str, set[str]] = defaultdict(set)
    evidence_by_hop: dict[tuple[str, str], list[str]] = defaultdict(list)

    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        relationship = edge["relationship_type"]
        evidence_by_hop[(source, target)].append(edge["edge_id"])
        if relationship in {"has_weakness", "observed_example_of"}:
            vuln_to_cwes[source].add(target)
        elif relationship in {"related_attack_pattern", "related_weakness_attack_pattern"}:
            if nodes[source]["entity_type"] == "cwe" and nodes[target]["entity_type"] == "capec":
                cwe_to_capecs[source].add(target)
        elif relationship == "maps_to_attack_technique":
            capec_to_attacks[source].add(target)
        elif relationship == "detected_by_sigma_rule":
            attack_to_sigmas[source].add(target)

    chains: list[dict[str, Any]] = []
    for cve_id in sorted(cve_to_nodes):
        source_vulnerability_nodes = sorted(cve_to_nodes[cve_id])
        nvd_node_id = nvd_by_cve.get(cve_id)
        kev_node_id = kev_by_cve.get(cve_id)
        primary_vulnerability_node_id = nvd_node_id or kev_node_id or source_vulnerability_nodes[0]

        cwe_node_ids: set[str] = set()
        for vuln_node_id in source_vulnerability_nodes:
            cwe_node_ids.update(vuln_to_cwes.get(vuln_node_id, set()))

        for cwe_node_id in sorted(cwe_node_ids):
            for capec_node_id in sorted(cwe_to_capecs.get(cwe_node_id, set())):
                for attack_node_id in sorted(capec_to_attacks.get(capec_node_id, set())):
                    chain_id = _stable_id(
                        "tier1-chain",
                        cve_id,
                        nodes[cwe_node_id]["entity_id"],
                        nodes[capec_node_id]["entity_id"],
                        nodes[attack_node_id]["entity_id"],
                    )
                    path_node_ids = [
                        primary_vulnerability_node_id,
                        cwe_node_id,
                        capec_node_id,
                        attack_node_id,
                    ]
                    path_relationships = [
                        "has_weakness",
                        "related_attack_pattern",
                        "maps_to_attack_technique",
                    ]
                    sigma_node_ids = sorted(attack_to_sigmas.get(attack_node_id, set()))
                    evidence_edge_ids = sorted(
                        set(
                            evidence_by_hop.get((vuln_node_id, cwe_node_id), [])[0]
                            for vuln_node_id in source_vulnerability_nodes
                            if evidence_by_hop.get((vuln_node_id, cwe_node_id))
                        )
                        | set(evidence_by_hop.get((cwe_node_id, capec_node_id), []))
                        | set(evidence_by_hop.get((capec_node_id, attack_node_id), []))
                    )
                    chain = {
                        "chain_id": chain_id,
                        "cve_id": cve_id,
                        "nvd_node_id": nvd_node_id,
                        "cisa_kev_node_id": kev_node_id,
                        "is_known_exploited": kev_node_id is not None,
                        "cwe_id": nodes[cwe_node_id]["entity_id"],
                        "cwe_title": nodes[cwe_node_id]["title"],
                        "capec_id": nodes[capec_node_id]["entity_id"],
                        "capec_title": nodes[capec_node_id]["title"],
                        "attack_technique_id": nodes[attack_node_id]["entity_id"],
                        "attack_technique_title": nodes[attack_node_id]["title"],
                        "attack_domains": nodes[attack_node_id]["attack_domains"],
                        "attack_tactics": nodes[attack_node_id]["attack_tactics"],
                        "sigma_rule_count": len(sigma_node_ids),
                        "path_node_ids": path_node_ids,
                        "path_relationships": path_relationships,
                        "evidence_edge_ids": evidence_edge_ids,
                        "chain_text": _format_chain_text(
                            cve_id=cve_id,
                            vulnerability_node=nodes[primary_vulnerability_node_id],
                            kev_node=nodes.get(kev_node_id) if kev_node_id else None,
                            cwe_node=nodes[cwe_node_id],
                            capec_node=nodes[capec_node_id],
                            attack_node=nodes[attack_node_id],
                        ),
                    }
                    chains.append(chain)
                    if max_chains is not None and len(chains) >= max_chains:
                        return chains
    return chains


def _extract_cwe_xml_relations(raw_xml: str) -> dict[str, Any]:
    root = _parse_xml(raw_xml)
    if root is None:
        return {"related_cwes": [], "related_capecs": [], "observed_cves": []}

    related_cwes = []
    for elem in root.xpath(".//*[local-name()='Related_Weakness']"):
        cwe_id = _normalize_cwe_id(elem.get("CWE_ID"))
        if cwe_id:
            related_cwes.append(
                {
                    "cwe_id": cwe_id,
                    "nature": elem.get("Nature"),
                    "view_id": elem.get("View_ID"),
                }
            )

    related_capecs = sorted(
        {
            capec_id
            for elem in root.xpath(".//*[local-name()='Related_Attack_Pattern']")
            for capec_id in [_normalize_capec_id(elem.get("CAPEC_ID"))]
            if capec_id
        }
    )

    observed_cves = sorted(
        {
            cve_id
            for elem in root.xpath(".//*[local-name()='Observed_Example']//*[local-name()='Reference']")
            for cve_id in _extract_cves("".join(elem.itertext()))
        }
    )
    return {
        "related_cwes": _dedupe_dicts(related_cwes, "cwe_id", "nature", "view_id"),
        "related_capecs": related_capecs,
        "observed_cves": observed_cves,
    }


def _extract_capec_xml_relations(raw_xml: str) -> dict[str, Any]:
    root = _parse_xml(raw_xml)
    if root is None:
        return {
            "related_cwes": [],
            "related_capecs": [],
            "attack_technique_ids": [],
            "example_cves": [],
        }

    related_cwes = sorted(
        {
            cwe_id
            for elem in root.xpath(".//*[local-name()='Related_Weakness']")
            for cwe_id in [_normalize_cwe_id(elem.get("CWE_ID"))]
            if cwe_id
        }
    )

    related_capecs = []
    for elem in root.xpath(".//*[local-name()='Related_Attack_Pattern']"):
        capec_id = _normalize_capec_id(elem.get("CAPEC_ID"))
        if capec_id:
            related_capecs.append({"capec_id": capec_id, "nature": elem.get("Nature")})

    attack_technique_ids = []
    for mapping in root.xpath(".//*[local-name()='Taxonomy_Mapping'][@Taxonomy_Name='ATTACK']"):
        for entry in mapping.xpath("./*[local-name()='Entry_ID']/text()"):
            attack_id = _normalize_attack_id(entry)
            if attack_id:
                attack_technique_ids.append(attack_id)

    example_text = "\n".join(
        "".join(elem.itertext()) for elem in root.xpath(".//*[local-name()='Example_Instances']")
    )
    return {
        "related_cwes": related_cwes,
        "related_capecs": _dedupe_dicts(related_capecs, "capec_id", "nature"),
        "attack_technique_ids": sorted(set(attack_technique_ids)),
        "example_cves": _extract_cves(example_text),
    }


def _iter_parquet_rows(paths: Iterable[Path], columns: list[str], *, batch_size: int = 8192) -> Iterator[dict[str, Any]]:
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        available_columns = [column for column in columns if column in parquet_file.schema_arrow.names]
        for batch in parquet_file.iter_batches(columns=available_columns, batch_size=batch_size):
            yield from batch.to_pylist()


def _write_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")


def _write_detection_chains(
    path: Path,
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    chains: list[dict[str, Any]],
    batch_size: int = 50_000,
) -> dict[str, int]:
    attack_to_sigma_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge["relationship_type"] == "detected_by_sigma_rule":
            attack_to_sigma_edges[edge["source_node_id"]].append(edge)

    rows_written = 0
    complete_kev_rows = 0
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None

    try:
        for chain in chains:
            attack_node_id = f"mitre-attack:{chain['attack_technique_id']}"
            for sigma_edge in sorted(attack_to_sigma_edges.get(attack_node_id, []), key=lambda item: item["target_node_id"]):
                sigma_node = nodes[sigma_edge["target_node_id"]]
                row = _make_detection_chain_row(chain, sigma_node, sigma_edge)
                batch.append(row)
                if row["cisa_kev_node_id"]:
                    complete_kev_rows += 1
                if len(batch) >= batch_size:
                    writer = _write_batch(path, batch, writer, _DETECTION_CHAIN_SCHEMA)
                    rows_written += len(batch)
                    batch = []

        if batch:
            writer = _write_batch(path, batch, writer, _DETECTION_CHAIN_SCHEMA)
            rows_written += len(batch)

        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=_DETECTION_CHAIN_SCHEMA), path, compression="snappy")
    finally:
        if writer is not None:
            writer.close()

    return {"rows": rows_written, "complete_kev_rows": complete_kev_rows}


def _write_batch(
    path: Path,
    rows: list[dict[str, Any]],
    writer: pq.ParquetWriter | None,
    schema: pa.Schema,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="snappy")
    writer.write_table(table)
    return writer


def _make_detection_chain_row(
    chain: dict[str, Any],
    sigma_node: dict[str, Any],
    sigma_edge: dict[str, Any],
) -> dict[str, Any]:
    path_node_ids = [*chain["path_node_ids"], sigma_node["node_id"]]
    path_relationships = [*chain["path_relationships"], "detected_by_sigma_rule"]
    evidence_edge_ids = sorted(set(chain["evidence_edge_ids"]) | {sigma_edge["edge_id"]})
    return {
        "detection_chain_id": _stable_id("tier1-detection-chain", chain["chain_id"], sigma_node["node_id"]),
        "base_chain_id": chain["chain_id"],
        "cve_id": chain["cve_id"],
        "nvd_node_id": chain["nvd_node_id"],
        "cisa_kev_node_id": chain["cisa_kev_node_id"],
        "is_known_exploited": chain["is_known_exploited"],
        "cwe_id": chain["cwe_id"],
        "cwe_title": chain["cwe_title"],
        "capec_id": chain["capec_id"],
        "capec_title": chain["capec_title"],
        "attack_technique_id": chain["attack_technique_id"],
        "attack_technique_title": chain["attack_technique_title"],
        "sigma_rule_id": sigma_node["entity_id"],
        "sigma_rule_title": sigma_node["title"],
        "sigma_rule_level": sigma_node["rule_level"],
        "sigma_node_id": sigma_node["node_id"],
        "sigma_source_url": sigma_node["source_url"],
        "path_node_ids": path_node_ids,
        "path_relationships": path_relationships,
        "evidence_edge_ids": evidence_edge_ids,
        "chain_text": _format_detection_chain_text(chain, sigma_node),
    }


def _decode_raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_xml(raw_xml: str) -> etree._Element | None:
    if not raw_xml:
        return None
    try:
        return etree.fromstring(raw_xml.encode("utf-8"), parser=_XML_PARSER)
    except etree.XMLSyntaxError:
        return None


def _normalize_cve_id(value: Any) -> str | None:
    if not value:
        return None
    match = _CVE_RE.search(str(value).upper())
    return match.group(0) if match else None


def _normalize_cwe_id(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().upper()
    if text.startswith("CWE-"):
        suffix = text.removeprefix("CWE-")
    else:
        suffix = text
    return f"CWE-{suffix}" if suffix.isdigit() else None


def _normalize_capec_id(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().upper()
    if text.startswith("CAPEC-"):
        suffix = text.removeprefix("CAPEC-")
    else:
        suffix = text
    return f"CAPEC-{suffix}" if suffix.isdigit() else None


def _normalize_attack_id(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().upper()
    if not text.startswith("T") and _ATTACK_ID_RE.fullmatch(text):
        text = f"T{text}"
    return text if _ATTACK_ID_RE.fullmatch(text) else None


def _normalize_id_list(values: Any, normalizer) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return sorted({normalized for value in values for normalized in [normalizer(value)] if normalized})


def _extract_cves(text: str) -> list[str]:
    return sorted({match.group(0).upper() for match in _CVE_RE.finditer(text or "")})


def _extract_sigma_attack_ids(rule_source: str) -> list[str]:
    return sorted(
        {
            attack_id
            for match in _SIGMA_ATTACK_TAG_RE.finditer(rule_source or "")
            for attack_id in [_normalize_attack_id(match.group(1))]
            if attack_id
        }
    )


def _as_string_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    if isinstance(values, list):
        return {str(value) for value in values if value}
    return {str(values)}


def _dedupe_dicts(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    deduped = {}
    for item in items:
        deduped[tuple(item.get(key) for key in keys)] = item
    return [deduped[key] for key in sorted(deduped)]


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:24]}"


def _format_chain_text(
    *,
    cve_id: str,
    vulnerability_node: dict[str, Any],
    kev_node: dict[str, Any] | None,
    cwe_node: dict[str, Any],
    capec_node: dict[str, Any],
    attack_node: dict[str, Any],
) -> str:
    parts = [
        (
            f"{cve_id} is described by {vulnerability_node['source_id']}: "
            f"{_snippet(vulnerability_node.get('content'), 360)}"
        )
    ]
    if kev_node:
        kev_title = _title(kev_node, "known exploited vulnerability")
        kev_summary = _snippet(kev_node.get("content"), 280)
        parts.append(f"CISA KEV marks {cve_id} as known exploited: {kev_title}. {kev_summary}")

    parts.extend(
        [
            (
                f"The CVE maps to {cwe_node['entity_id']} ({_title(cwe_node, 'CWE entry')}): "
                f"{_snippet(cwe_node.get('content'), 320)}"
            ),
            (
                f"{cwe_node['entity_id']} is linked to {capec_node['entity_id']} "
                f"({_title(capec_node, 'CAPEC pattern')}): {_snippet(capec_node.get('content'), 320)}"
            ),
            (
                f"{capec_node['entity_id']} maps to MITRE ATT&CK {attack_node['entity_id']} "
                f"({_title(attack_node, 'ATT&CK technique')}): {_snippet(attack_node.get('content'), 360)}"
            ),
        ]
    )
    return "\n".join(parts)


def _format_detection_chain_text(chain: dict[str, Any], sigma_node: dict[str, Any]) -> str:
    sigma_summary = _snippet(_sigma_description(sigma_node.get("content")), 280)
    level = f", level {sigma_node['rule_level']}" if sigma_node.get("rule_level") else ""
    return (
        f"{chain['chain_text']}\n"
        f"Sigma rule {sigma_node['entity_id']} ({_title(sigma_node, 'Sigma rule')}{level}) "
        f"is tagged to detect {chain['attack_technique_id']}: {sigma_summary}"
    )


def _title(node: dict[str, Any], fallback: str) -> str:
    return node.get("title") or fallback


def _snippet(text: Any, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return "No source description available."
    if len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{trimmed}..."


def _sigma_description(content: Any) -> str:
    text = str(content or "")
    if "```yaml" in text:
        text = text.split("```yaml", 1)[0]
    return text.strip() or content or ""
