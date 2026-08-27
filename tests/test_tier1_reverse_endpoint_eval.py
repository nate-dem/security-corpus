from ingest.derived.tier1_reverse_endpoint_eval import (
    extract_first_json_object,
    parse_model_response,
    score_prediction,
    summarize_prediction_rows,
)


def test_score_prediction_awards_exact_chain_points():
    example = _example()
    parsed = {
        "cve_id": "CVE-2024-0001",
        "chain": {
            "sigma_rule": "sigma:SIGMA-1",
            "attack_technique": "mitre-attack:T1000",
            "capec": "capec:CAPEC-1",
            "cwe": "mitre-cwe:CWE-1",
            "cve": "nvd:CVE-2024-0001",
        },
    }

    score = score_prediction(parsed, example)

    assert score["points"] == 5
    assert score["score"] == 1.0
    assert score["cve_id_correct"] is True
    assert score["exact_chain_match"] is True
    assert all(score["hop_correct"].values())
    assert all(score["field_correct"].values())


def test_score_prediction_allows_partial_valid_distractor_path_credit():
    example = _example()
    parsed = {
        "cve_id": "CVE-2024-0002",
        "chain": {
            "sigma_rule": "sigma:SIGMA-2",
            "attack_technique": "mitre-attack:T2000",
            "capec": "capec:CAPEC-2",
            "cwe": "mitre-cwe:CWE-2",
            "cve": "nvd:CVE-2024-0002",
        },
    }

    score = score_prediction(parsed, example)

    assert score["points"] == 4
    assert score["cve_id_correct"] is False
    assert score["exact_chain_match"] is False
    assert all(score["hop_correct"].values())
    assert not any(score["field_correct"].values())


def test_score_prediction_handles_parse_failure():
    score = score_prediction(None, _example())

    assert score["points"] == 0
    assert score["cve_id_correct"] is False
    assert score["exact_chain_match"] is False


def test_parse_model_response_extracts_first_json_object():
    parsed, status = parse_model_response(
        '```json\n{"cve_id":"CVE-2024-0001","chain":{"cve":"nvd:CVE-2024-0001"}}\n```'
    )

    assert status == "ok"
    assert parsed["cve_id"] == "CVE-2024-0001"
    assert extract_first_json_object("no json") is None


def test_summarize_prediction_rows():
    rows = [
        {
            "parse_status": "ok",
            "exact_chain_match": True,
            "cve_id_correct": True,
            "points": 5,
            "max_points": 5,
            "hop_correct": {"cve_to_cwe": True},
            "field_correct": {"cve": True},
        },
        {
            "parse_status": "no_json_object",
            "exact_chain_match": False,
            "cve_id_correct": False,
            "points": 0,
            "max_points": 5,
            "hop_correct": {"cve_to_cwe": False},
            "field_correct": {"cve": False},
        },
    ]

    summary = summarize_prediction_rows(
        rows,
        model="test-model",
        input_path="input.jsonl",
        output_path="output.jsonl",
    )

    assert summary["examples"] == 2
    assert summary["parse_failures"] == 1
    assert summary["exact_chain_accuracy"] == 0.5
    assert summary["partial_credit_accuracy"] == 0.5
    assert summary["hop_accuracy"]["cve_to_cwe"]["accuracy"] == 0.5


def _example():
    return {
        "benchmark_id": "bench:test",
        "answer": {
            "cve_id": "CVE-2024-0001",
            "chain": {
                "sigma_rule": "sigma:SIGMA-1",
                "attack_technique": "mitre-attack:T1000",
                "capec": "capec:CAPEC-1",
                "cwe": "mitre-cwe:CWE-1",
                "cve": "nvd:CVE-2024-0001",
            },
        },
        "input": {
            "edges": [
                {"source": "nvd:CVE-2024-0001", "relationship": "has_weakness", "target": "mitre-cwe:CWE-1"},
                {"source": "mitre-cwe:CWE-1", "relationship": "related_attack_pattern", "target": "capec:CAPEC-1"},
                {"source": "capec:CAPEC-1", "relationship": "maps_to_attack_technique", "target": "mitre-attack:T1000"},
                {"source": "mitre-attack:T1000", "relationship": "detected_by_sigma_rule", "target": "sigma:SIGMA-1"},
                {"source": "nvd:CVE-2024-0002", "relationship": "has_weakness", "target": "mitre-cwe:CWE-2"},
                {"source": "mitre-cwe:CWE-2", "relationship": "related_attack_pattern", "target": "capec:CAPEC-2"},
                {"source": "capec:CAPEC-2", "relationship": "maps_to_attack_technique", "target": "mitre-attack:T2000"},
                {"source": "mitre-attack:T2000", "relationship": "detected_by_sigma_rule", "target": "sigma:SIGMA-2"},
            ]
        },
    }
