from datetime import datetime, timezone
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.release import baseline_preflight


def project_fixture(root, missing_url=False):
    (root/'config').mkdir()
    (root/'data').mkdir()
    text='Virtual memory separates process address spaces.'
    row={'source_id':'example','source_record_id':'1','record_id':'example:1',
        'content':text,'content_hash':compute_content_hash(text),'content_length':compute_token_count(text),
        'license':'CC-BY-4.0','source_url':'' if missing_url else 'https://example.org/1',
        'ingested_at':datetime.now(timezone.utc)}
    pq.write_table(pa.Table.from_pylist([row]),root/'data/retained.parquet')
    (root/'config/retained_baseline.json').write_text(json.dumps({'inputs':['data/retained.parquet'],
        'expected_records':1,'reference_stored_tokens':row['content_length']}))
    (root/'config/source_licenses.yaml').write_text('schema_version: 1\npolicies:\n  - source_patterns: [example]\n    state: conditional\n    expected_licenses: [CC-BY-4.0]\n')
    return row


@pytest.mark.parametrize('missing_url',[False,True])
def test_completed_inventory_reports_findings_without_claiming_release_approval(tmp_path,missing_url):
    row=project_fixture(tmp_path,missing_url)
    output=tmp_path/'review'
    before=(tmp_path/'data/retained.parquet').read_bytes()
    args=['--project-root',str(tmp_path),'--output-dir',str(output),'--workers','1']
    assert baseline_preflight.main(args)==0
    summary=json.loads((output/'summary.json').read_text())
    assert summary['scan_complete'] and not summary['release_ready']
    assert summary['integrity_passed'] is (not missing_url)
    assert summary['recomputed_tokens']==row['content_length']
    manifest=json.loads((output/'manifest.draft.json').read_text())
    assert manifest['aspirational_target_tokens']==3_000_000_000
    assert manifest['target_is_release_gate'] is False
    assert manifest['license_inventory']['state_totals']['conditional']['records']==1
    assert manifest['files'][0]['project_relative_path']=='data/retained.parquet'
    assert before==(tmp_path/'data/retained.parquet').read_bytes()
    with pytest.raises(FileExistsError,match='Preserve completed'):
        baseline_preflight.main(args)
