import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.derived.tier1_links import (
    _extract_capec_xml_relations,
    _extract_cwe_xml_relations,
    _extract_sigma_attack_ids,
    build_tier1_reasoning_dataset,
)


def test_extracts_cwe_and_capec_xml_relationships():
    cwe_xml = """
    <Weakness xmlns="http://cwe.mitre.org/cwe-7" ID="79" Name="XSS">
      <Observed_Examples>
        <Observed_Example><Reference>CVE-2024-12345</Reference></Observed_Example>
      </Observed_Examples>
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="74" View_ID="1000"/>
      </Related_Weaknesses>
      <Related_Attack_Patterns>
        <Related_Attack_Pattern CAPEC_ID="63"/>
      </Related_Attack_Patterns>
    </Weakness>
    """
    capec_xml = """
    <Attack_Pattern xmlns="http://capec.mitre.org/capec-3" ID="63" Name="XSS">
      <Related_Weaknesses>
        <Related_Weakness CWE_ID="79"/>
      </Related_Weaknesses>
      <Related_Attack_Patterns>
        <Related_Attack_Pattern Nature="ChildOf" CAPEC_ID="242"/>
      </Related_Attack_Patterns>
      <Taxonomy_Mappings>
        <Taxonomy_Mapping Taxonomy_Name="ATTACK">
          <Entry_ID>1059.007</Entry_ID>
        </Taxonomy_Mapping>
      </Taxonomy_Mappings>
      <Example_Instances>
        <Example>CVE-2024-54321</Example>
      </Example_Instances>
    </Attack_Pattern>
    """

    assert _extract_cwe_xml_relations(cwe_xml) == {
        "related_cwes": [{"cwe_id": "CWE-74", "nature": "ChildOf", "view_id": "1000"}],
        "related_capecs": ["CAPEC-63"],
        "observed_cves": ["CVE-2024-12345"],
    }
    assert _extract_capec_xml_relations(capec_xml) == {
        "related_cwes": ["CWE-79"],
        "related_capecs": [{"capec_id": "CAPEC-242", "nature": "ChildOf"}],
        "attack_technique_ids": ["T1059.007"],
        "example_cves": ["CVE-2024-54321"],
    }
    assert _extract_sigma_attack_ids("tags:\n- attack.t1059.007\n- attack.execution\n") == ["T1059.007"]


def test_build_tier1_reasoning_dataset(tmp_path: Path):
    data_dir = tmp_path / "data" / "training-clean-v2" / "normalized"
    output_dir = tmp_path / "data" / "tier1-reasoning-clean-v2"
    _write_source(
        data_dir,
        "source_id=nvd/part-00000.parquet",
        [
            {
                "source_id": "nvd",
                "source_record_id": "CVE-2024-0001",
                "record_id": "nvd:CVE-2024-0001",
                "content": "A test vulnerability.",
                "title": None,
                "content_length": 4,
                "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
                "license": "public-domain",
                "raw": None,
                "cve_id": "CVE-2024-0001",
                "severity": "high",
                "cvss_score": 9.8,
                "cwe_ids": ["CWE-79"],
                "exploited_in_wild": None,
            }
        ],
    )
    _write_source(
        data_dir,
        "source_id=cisa-kev/part-00000.parquet",
        [
            {
                "source_id": "cisa-kev",
                "source_record_id": "CVE-2024-0001",
                "record_id": "cisa-kev:CVE-2024-0001",
                "content": "Known exploited test vulnerability.",
                "title": "Vendor Product XSS",
                "content_length": 5,
                "source_url": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                "license": "cisa-terms",
                "raw": None,
                "cve_id": "CVE-2024-0001",
                "severity": None,
                "cvss_score": None,
                "cwe_ids": ["CWE-79"],
                "exploited_in_wild": True,
            }
        ],
    )
    _write_source(
        data_dir,
        "source_id=mitre-cwe/part-00000.parquet",
        [
            {
                "source_id": "mitre-cwe",
                "source_record_id": "CWE-79",
                "record_id": "mitre-cwe:CWE-79",
                "content": "Improper neutralization.",
                "title": "Cross-site Scripting",
                "content_length": 3,
                "source_url": "https://cwe.mitre.org/data/definitions/79.html",
                "license": "mitre-terms",
                "raw": json.dumps(
                    {
                        "raw_xml": """
                        <Weakness xmlns="http://cwe.mitre.org/cwe-7" ID="79" Name="XSS">
                          <Observed_Examples>
                            <Observed_Example><Reference>CVE-2024-0001</Reference></Observed_Example>
                          </Observed_Examples>
                          <Related_Attack_Patterns>
                            <Related_Attack_Pattern CAPEC_ID="63"/>
                          </Related_Attack_Patterns>
                        </Weakness>
                        """
                    }
                ),
                "category_id": "CWE-79",
            }
        ],
    )
    _write_source(
        data_dir,
        "source_id=capec/part-00000.parquet",
        [
            {
                "source_id": "capec",
                "source_record_id": "CAPEC-63",
                "record_id": "capec:CAPEC-63",
                "content": "Cross-site scripting attack pattern.",
                "title": "Cross-Site Scripting",
                "content_length": 4,
                "source_url": "https://capec.mitre.org/data/definitions/63.html",
                "license": "mitre-terms",
                "raw": json.dumps(
                    {
                        "raw_xml": """
                        <Attack_Pattern xmlns="http://capec.mitre.org/capec-3" ID="63" Name="XSS">
                          <Related_Weaknesses>
                            <Related_Weakness CWE_ID="79"/>
                          </Related_Weaknesses>
                          <Taxonomy_Mappings>
                            <Taxonomy_Mapping Taxonomy_Name="ATTACK">
                              <Entry_ID>1059.007</Entry_ID>
                            </Taxonomy_Mapping>
                          </Taxonomy_Mappings>
                        </Attack_Pattern>
                        """
                    }
                ),
                "category_id": "CAPEC-63",
            }
        ],
    )
    _write_source(
        data_dir,
        "source_id=mitre-attack/part-00000.parquet",
        [
            {
                "source_id": "mitre-attack",
                "source_record_id": "T1059.007",
                "record_id": "mitre-attack:T1059.007",
                "content": "JavaScript execution.",
                "title": "JavaScript",
                "content_length": 2,
                "source_url": "https://attack.mitre.org/techniques/T1059/007",
                "license": "mitre-terms",
                "raw": json.dumps(
                    {
                        "type": "attack-pattern",
                        "x_mitre_domains": ["enterprise-attack"],
                        "kill_chain_phases": [
                            {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                        ],
                    }
                ),
                "category_id": "T1059.007",
            }
        ],
    )
    _write_source(
        data_dir,
        "source_id=sigma/part-00000.parquet",
        [
            {
                "source_id": "sigma",
                "source_record_id": "39c9f26d-6e3b-4dbb-9c7a-4154b0281112",
                "record_id": "sigma:39c9f26d-6e3b-4dbb-9c7a-4154b0281112",
                "content": "Detects JavaScript execution.\n\n```yaml\ntags:\n- attack.t1059.007\n```",
                "title": "JavaScript Execution",
                "content_length": 8,
                "source_url": "https://github.com/SigmaHQ/sigma/blob/master/rules/test.yml",
                "license": "lgpl-2.1",
                "raw": None,
                "rule_id": "39c9f26d-6e3b-4dbb-9c7a-4154b0281112",
                "rule_level": "medium",
                "rule_source": "tags:\n- attack.t1059.007\n",
            }
        ],
    )

    result = build_tier1_reasoning_dataset(data_dir, output_dir)

    assert result.nodes == 6
    assert result.chains == 1
    assert result.chains_with_sigma == 1
    assert result.detection_chains == 1
    assert result.complete_kev_chains == 1
    assert result.complete_kev_detection_chains == 1

    edges = pq.read_table(output_dir / "edges.parquet").to_pylist()
    relationships = {edge["relationship_type"] for edge in edges}
    assert "same_cve" in relationships
    assert "has_weakness" in relationships
    assert "related_attack_pattern" in relationships
    assert "related_weakness_attack_pattern" in relationships
    assert "maps_to_attack_technique" in relationships
    assert "detected_by_sigma_rule" in relationships

    chain = pq.read_table(output_dir / "chains.parquet").to_pylist()[0]
    assert chain["cve_id"] == "CVE-2024-0001"
    assert chain["cwe_id"] == "CWE-79"
    assert chain["capec_id"] == "CAPEC-63"
    assert chain["attack_technique_id"] == "T1059.007"
    assert chain["is_known_exploited"] is True
    assert chain["sigma_rule_count"] == 1
    assert "CVE-2024-0001 is described by nvd: A test vulnerability." in chain["chain_text"]
    assert "CISA KEV marks CVE-2024-0001 as known exploited" in chain["chain_text"]
    assert "The CVE maps to CWE-79 (Cross-site Scripting): Improper neutralization." in chain["chain_text"]
    assert "CAPEC-63 (Cross-Site Scripting): Cross-site scripting attack pattern." in chain["chain_text"]
    assert "MITRE ATT&CK T1059.007 (JavaScript): JavaScript execution." in chain["chain_text"]

    detection_chain = pq.read_table(output_dir / "detection_chains.parquet").to_pylist()[0]
    assert detection_chain["cve_id"] == "CVE-2024-0001"
    assert detection_chain["attack_technique_id"] == "T1059.007"
    assert detection_chain["sigma_rule_id"] == "39c9f26d-6e3b-4dbb-9c7a-4154b0281112"
    assert detection_chain["sigma_rule_level"] == "medium"
    assert "Sigma rule 39c9f26d-6e3b-4dbb-9c7a-4154b0281112" in detection_chain["chain_text"]
    assert "Detects JavaScript execution." in detection_chain["chain_text"]
    assert detection_chain["path_relationships"] == [
        "has_weakness",
        "related_attack_pattern",
        "maps_to_attack_technique",
        "detected_by_sigma_rule",
    ]


def test_rejects_non_clean_v2_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="clean-v2 normalized dataset"):
        build_tier1_reasoning_dataset(
            tmp_path / "data",
            tmp_path / "data" / "tier1-reasoning-clean-v2",
        )

    with pytest.raises(ValueError, match="clean-v2 output directory"):
        build_tier1_reasoning_dataset(
            tmp_path / "data" / "training-clean-v2" / "normalized",
            tmp_path / "data" / "tier1-reasoning",
        )


def _write_source(root: Path, relative_path: str, rows: list[dict]):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
