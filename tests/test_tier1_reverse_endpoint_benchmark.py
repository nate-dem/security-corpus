import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.derived import tier1_reverse_endpoint_benchmark as benchmark_module
from ingest.derived.tier1_reverse_endpoint_benchmark import build_reverse_endpoint_benchmark


def test_reverse_endpoint_benchmark_builds_verified_graph_task(tmp_path: Path):
    benchmark_module._TOKEN_ENCODER = _FakeTokenEncoder()
    priority_path = tmp_path / "priority_chains.parquet"
    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    output_path = tmp_path / "benchmark.jsonl"

    pq.write_table(
        pa.Table.from_pylist(
            [
                _priority_row(
                    cve_id="CVE-2024-0001",
                    cwe_id="CWE-1",
                    capec_id="CAPEC-1",
                    attack_id="T1000",
                    sigma_id="SIGMA-1",
                    edge_ids=["e-gold-1", "e-gold-2", "e-gold-3", "e-gold-4"],
                    priority=1,
                )
            ]
        ),
        priority_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _node("nvd:CVE-2024-0001", "cve", "CVE-2024-0001", content="Vendor product allows privilege escalation through unsafe input handling."),
                _node("mitre-cwe:CWE-1", "cwe", "CWE-1", title="Gold CWE"),
                _node("capec:CAPEC-1", "capec", "CAPEC-1", title="Gold CAPEC"),
                _node("mitre-attack:T1000", "attack-technique", "T1000", title="Gold Attack"),
                _node("sigma:SIGMA-1", "sigma-rule", "SIGMA-1", title="Gold Sigma", rule_level="high"),
                _node("nvd:CVE-2024-0002", "cve", "CVE-2024-0002", content="Distractor vulnerability."),
                _node("mitre-cwe:CWE-2", "cwe", "CWE-2", title="Related CWE"),
                _node("capec:CAPEC-2", "capec", "CAPEC-2", title="Wrong CAPEC"),
                _node("mitre-attack:T2000", "attack-technique", "T2000", title="Wrong Attack"),
                _node("sigma:SIGMA-2", "sigma-rule", "SIGMA-2", title="Wrong Sigma", rule_level="low"),
            ]
        ),
        nodes_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _edge("e-gold-1", "nvd:CVE-2024-0001", "mitre-cwe:CWE-1", "has_weakness", "cve", "CVE-2024-0001", "cwe", "CWE-1"),
                _edge("e-gold-2", "mitre-cwe:CWE-1", "capec:CAPEC-1", "related_attack_pattern", "cwe", "CWE-1", "capec", "CAPEC-1"),
                _edge("e-gold-3", "capec:CAPEC-1", "mitre-attack:T1000", "maps_to_attack_technique", "capec", "CAPEC-1", "attack-technique", "T1000"),
                _edge("e-gold-4", "mitre-attack:T1000", "sigma:SIGMA-1", "detected_by_sigma_rule", "attack-technique", "T1000", "sigma-rule", "SIGMA-1"),
                _edge("e-related", "mitre-cwe:CWE-1", "mitre-cwe:CWE-2", "related_weakness", "cwe", "CWE-1", "cwe", "CWE-2"),
                _edge("e-wrong-1", "nvd:CVE-2024-0002", "mitre-cwe:CWE-2", "has_weakness", "cve", "CVE-2024-0002", "cwe", "CWE-2"),
                _edge("e-wrong-2", "mitre-cwe:CWE-2", "capec:CAPEC-2", "related_attack_pattern", "cwe", "CWE-2", "capec", "CAPEC-2"),
                _edge("e-wrong-3", "capec:CAPEC-2", "mitre-attack:T2000", "maps_to_attack_technique", "capec", "CAPEC-2", "attack-technique", "T2000"),
                _edge("e-wrong-4", "mitre-attack:T2000", "sigma:SIGMA-2", "detected_by_sigma_rule", "attack-technique", "T2000", "sigma-rule", "SIGMA-2"),
            ]
        ),
        edges_path,
    )

    summary = build_reverse_endpoint_benchmark(
        priority_path,
        output_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
        max_examples=1,
        seed=7,
        batch_size=2,
        priorities=(1,),
        min_distractor_nodes=5,
    )

    assert summary.examples_written == 1
    example = json.loads(output_path.read_text().strip())
    assert example["task_type"] == "tier1_sigma_to_cve_path_json"
    assert example["answer"]["cve_id"] == "CVE-2024-0001"
    assert example["answer"]["path_edge_ids"] == ["e-gold-1", "e-gold-2", "e-gold-3", "e-gold-4"]
    assert "edges" not in example["answer"]["chain"]
    assert example["output_json_schema"]["properties"]["chain"]["required"] == [
        "sigma_rule",
        "attack_technique",
        "capec",
        "cwe",
        "cve",
    ]
    assert example["input"]["target_sigma_rule"]["rule_id"] == "SIGMA-1"
    assert "source_url" not in example["input"]["target_sigma_rule"]
    assert "CVE-2024-0001" not in example["input"]["target_cve_description"]
    assert len(example["input"]["nodes"]) == 10
    assert len(example["input"]["edges"]) == 8
    assert all("source_url" not in node for node in example["input"]["nodes"])
    assert all("evidence" not in edge for edge in example["input"]["edges"])
    assert all("edge_id" not in edge for edge in example["input"]["edges"])
    assert example["verification"]["gold_hops_verified"] is True
    assert example["verification"]["exactly_one_valid_path_to_target_sigma"] is True
    assert example["verification"]["distractor_node_count"] == 5
    assert example["scoring"]["partial_credit"]["cve_to_cwe"] == 1
    assert set(example["metadata"]["distractor_types"]) == {"related_cwe_wrong_path"}
    assert example["metadata"]["synthetic_distractor_openai_chain_count"] == 0
    assert example["metadata"]["synthetic_distractor_local_template_chain_count"] == 0

    model_prompt = benchmark_module.assemble_reverse_endpoint_model_prompt(example)
    assert "Provided graph input:" in model_prompt
    assert "Required output JSON schema:" in model_prompt
    assert "nvd:CVE-2024-0001" in model_prompt
    assert "mitre-cwe:CWE-1" in model_prompt
    assert "path_edge_ids" not in model_prompt
    assert "source_target_cve_description" not in model_prompt
    assert "synthetic_distractor_openai_chain_count" not in model_prompt
    assert example["metadata"]["model_prompt_token_count"] == benchmark_module._count_tokens(model_prompt)


def test_synthetic_distractor_graph_builds_false_chain_without_source_ids():
    nodes, edges, chain_count = benchmark_module._synthetic_distractor_graph(
        benchmark_id="benchmark:test",
        distractors=[
            {
                "cve_description": "A near-miss CVE-2024-9999 SQL injection description.",
                "cwe_title": "Improper Query Neutralization",
                "cwe_description": "Inputs are not neutralized before database query construction.",
                "capec_title": "Manipulate Query Parameters",
                "capec_description": "An attacker changes request parameters to alter query behavior.",
                "attack_technique_title": "Obfuscated Command Execution",
                "attack_technique_description": "Execution is hidden through indirect command paths.",
                "sigma_rule_title": "Suspicious Query Tool Launch",
                "sigma_rule_description": "Detects command-line patterns associated with suspicious query tooling.",
            }
        ],
    )

    assert chain_count == 1
    assert len(nodes) == 5
    assert len(edges) == 4
    assert [edge["relationship"] for edge in edges] == [
        "has_weakness",
        "related_attack_pattern",
        "maps_to_attack_technique",
        "detected_by_sigma_rule",
    ]
    assert all(edge["target"] != "sigma:SIGMA-1" for edge in edges)
    assert all("source_url" not in node for node in nodes)
    assert all("evidence" not in edge for edge in edges)
    assert all("edge_id" not in edge for edge in edges)
    node_ids = {node["node_id"] for node in nodes}
    assert all(not node_id.startswith(("mitre-cwe:CWE-9", "capec:CAPEC-9", "mitre-attack:T9")) for node_id in node_ids)
    assert not any(node_id.startswith("nvd:CVE-2024-9") and len(node_id.rsplit("-", 1)[-1]) == 6 for node_id in node_ids)
    assert "[REDACTED_CVE_ID]" in nodes[0]["description"]
    assert "CVE-2024-9999" not in json.dumps(nodes)
    assert "synthetic" not in json.dumps(nodes).lower()


def test_generated_meta_terms_are_stripped_from_synthetic_nodes():
    nodes, _, chain_count = benchmark_module._synthetic_distractor_graph(
        benchmark_id="benchmark:test-meta-terms",
        distractors=[
            {
                "cve_description": "A synthetic plugin exposes a fake debug endpoint.",
                "cwe": "Synthetic weakness: exposed diagnostic mode.",
                "capec": "Synthetic pattern: debug endpoint probing.",
                "attack_technique": "Synthetic technique: probing exposed diagnostics.",
                "sigma_rule": "Synthetic rule: debug endpoint scan.",
            }
        ],
    )

    assert chain_count == 1
    node_text = json.dumps(nodes).lower()
    assert "synthetic" not in node_text
    assert "fake " not in node_text


def test_local_synthetic_distractors_are_usable_without_meta_terms():
    distractors = benchmark_module._local_synthetic_distractors(
        benchmark_id="benchmark:local-fallback",
        target_cve_description="A parser library mishandles crafted profile metadata before version 1.2.3.",
        target_sigma_rule={"title": "Suspicious Encoded Loader", "node_id": "sigma:test"},
        count=3,
        batch_index=1,
    )
    nodes, edges, chain_count = benchmark_module._synthetic_distractor_graph(
        benchmark_id="benchmark:local-fallback",
        distractors=distractors,
    )

    assert chain_count == 3
    assert len(nodes) == 15
    assert len(edges) == 12
    node_text = json.dumps(nodes).lower()
    for meta_term in ("synthetic", "distractor", "decoy", "fake", "placeholder", "dummy"):
        assert meta_term not in node_text


def test_diversity_cap_check_does_not_mutate_counts():
    attack_counts: dict[str, int] = {}
    capec_counts: dict[str, int] = {}
    cwe_counts: dict[str, int] = {}
    cve_counts: dict[str, int] = {}
    gold_row = {
        "attack_technique_id": "T1000",
        "capec_id": "CAPEC-1",
        "cwe_id": "CWE-1",
        "cve_id": "CVE-2024-0001",
    }

    assert benchmark_module._within_diversity_caps(
        gold_row,
        attack_counts=attack_counts,
        capec_counts=capec_counts,
        cwe_counts=cwe_counts,
        cve_counts=cve_counts,
        max_examples_per_attack_technique=1,
        max_examples_per_capec=1,
        max_examples_per_cwe=1,
        max_examples_per_cve=1,
    )
    assert attack_counts == {}
    assert capec_counts == {}
    assert cwe_counts == {}
    assert cve_counts == {}

    cve_counts["CVE-2024-0001"] = 1
    assert not benchmark_module._within_diversity_caps(
        gold_row,
        attack_counts=attack_counts,
        capec_counts=capec_counts,
        cwe_counts=cwe_counts,
        cve_counts=cve_counts,
        max_examples_per_attack_technique=1,
        max_examples_per_capec=1,
        max_examples_per_cwe=1,
        max_examples_per_cve=1,
    )


def test_nvd_endpoint_filter_rejects_cisa_kev_endpoint():
    assert benchmark_module._has_nvd_cve_endpoint(
        {
            "nvd_node_id": None,
            "path_node_ids": ["cisa-kev:CVE-2020-11261", "mitre-cwe:CWE-20"],
        }
    ) is False
    assert benchmark_module._has_nvd_cve_endpoint(
        {
            "nvd_node_id": "nvd:CVE-2020-11261",
            "path_node_ids": ["cisa-kev:CVE-2020-11261", "mitre-cwe:CWE-20"],
        }
    ) is False
    assert benchmark_module._has_nvd_cve_endpoint(
        {
            "nvd_node_id": "nvd:CVE-2020-11261",
            "path_node_ids": ["nvd:CVE-2020-11261", "mitre-cwe:CWE-20"],
        }
    ) is True


def test_hard_prediction_profile_scores_failed_neighborhoods(tmp_path: Path):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "exact_chain_match": False,
                "points": 3,
                "gold_cve_id": "CVE-2024-0001",
                "predicted_cve_id": "CVE-2024-9999",
                "gold_chain": {
                    "sigma_rule": "sigma:SIGMA-1",
                    "attack_technique": "mitre-attack:T1000",
                    "capec": "capec:CAPEC-1",
                    "cwe": "mitre-cwe:CWE-1",
                    "cve": "nvd:CVE-2024-0001",
                },
                "predicted_chain": {
                    "sigma_rule": "sigma:SIGMA-1",
                    "attack_technique": "mitre-attack:T2000",
                    "capec": "capec:CAPEC-2",
                    "cwe": "mitre-cwe:CWE-2",
                    "cve": "nvd:CVE-2024-9999",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    profile = benchmark_module._load_hard_selection_profile(
        predictions_path,
        min_selection_score=1,
        max_points=None,
    )
    hard_row = _priority_row(
        cve_id="CVE-2024-0003",
        cwe_id="CWE-1",
        capec_id="CAPEC-2",
        attack_id="T2000",
        sigma_id="SIGMA-3",
        edge_ids=["e1", "e2", "e3", "e4"],
        priority=1,
    )
    unrelated_row = _priority_row(
        cve_id="CVE-2024-0004",
        cwe_id="CWE-44",
        capec_id="CAPEC-44",
        attack_id="T4444",
        sigma_id="SIGMA-4",
        edge_ids=["e1", "e2", "e3", "e4"],
        priority=1,
    )

    assert profile.failure_count == 1
    assert benchmark_module._hard_selection_score(hard_row, profile) == 20
    assert benchmark_module._hard_selection_score(unrelated_row, profile) == 0


def _priority_row(
    *,
    cve_id: str,
    cwe_id: str,
    capec_id: str,
    attack_id: str,
    sigma_id: str,
    edge_ids: list[str],
    priority: int,
):
    return {
        "priority": priority,
        "chain_record_id": f"chain:{cve_id}:{sigma_id}",
        "base_chain_id": f"base:{cve_id}:{sigma_id}",
        "detection_chain_id": f"detection:{cve_id}:{sigma_id}",
        "cve_id": cve_id,
        "nvd_node_id": f"nvd:{cve_id}",
        "cwe_id": cwe_id,
        "cwe_title": f"{cwe_id} title",
        "capec_id": capec_id,
        "capec_title": f"{capec_id} title",
        "attack_technique_id": attack_id,
        "attack_technique_title": f"{attack_id} title",
        "sigma_rule_id": sigma_id,
        "sigma_rule_title": f"{sigma_id} title",
        "sigma_rule_level": "medium",
        "sigma_node_id": f"sigma:{sigma_id}",
        "sigma_source_url": f"https://example.test/{sigma_id}",
        "path_node_ids": [
            f"nvd:{cve_id}",
            f"mitre-cwe:{cwe_id}",
            f"capec:{capec_id}",
            f"mitre-attack:{attack_id}",
            f"sigma:{sigma_id}",
        ],
        "path_relationships": [
            "has_weakness",
            "related_attack_pattern",
            "maps_to_attack_technique",
            "detected_by_sigma_rule",
        ],
        "evidence_edge_ids": edge_ids,
        "content_length": 100,
    }


def _node(
    node_id: str,
    entity_type: str,
    entity_id: str,
    *,
    title: str | None = None,
    content: str | None = None,
    rule_level: str | None = None,
):
    return {
        "node_id": node_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source_id": entity_type,
        "source_record_id": entity_id,
        "record_id": node_id,
        "title": title,
        "content": content or f"{entity_id} description.",
        "content_length": 10,
        "source_url": f"https://example.test/{entity_id}",
        "license": "test",
        "severity": None,
        "cvss_score": None,
        "exploited_in_wild": None,
        "attack_domains": [],
        "attack_tactics": [],
        "rule_level": rule_level,
        "sigma_attack_tags": [],
    }


def _edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    relationship_type: str,
    source_entity_type: str,
    source_entity_id: str,
    target_entity_type: str,
    target_entity_id: str,
):
    return {
        "edge_id": edge_id,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "relationship_type": relationship_type,
        "source_entity_type": source_entity_type,
        "source_entity_id": source_entity_id,
        "target_entity_type": target_entity_type,
        "target_entity_id": target_entity_id,
        "evidence_source_id": "test",
        "evidence_record_id": source_node_id,
        "evidence_field": "fixture",
        "evidence_detail": target_node_id,
    }


class _FakeTokenEncoder:
    def encode(self, text: str) -> list[str]:
        return text.split()
