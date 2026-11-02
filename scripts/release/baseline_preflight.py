"""Read-only baseline integrity/terms inventory and draft assembly manifest.

Exit 0 means the scan completed, including when it found release blockers. No
source is removed, no files are published, and no legal permission is inferred.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from . import audit_corpus_integrity as integrity, audit_source_licenses as licenses


def draft_manifest(project, audit, policy, config):
    files=[Path(row['path']) for row in audit['files']]
    if any(not p.resolve().is_relative_to(project.resolve()) for p in files):
        raise ValueError('Baseline audit includes a file outside the project')
    license_rows=licenses._summarize(files)
    license_report=licenses._build_report(license_rows,policy,files)
    return {'schema_version':1,'status':'draft_inventory','release_ready':False,
        'created_at':datetime.now(timezone.utc).isoformat(),
        'aspirational_target_tokens':3_000_000_000,'target_is_release_gate':False,
        'source_selection_changed':False,'integrity_passed':audit['integrity_passed'],
        'records':audit['records'],'stored_tokens':audit['stored_tokens'],'recomputed_tokens':audit['recomputed_tokens'],
        'matches_reference_inventory':audit['records']==config['expected_records'] and audit['stored_tokens']==config['reference_stored_tokens'],
        'integrity_issues':audit['issues'],'exact_duplicates':audit['duplicates'],
        'license_inventory':license_report,
        'files':[{'project_relative_path':str(Path(r['path']).resolve().relative_to(project.resolve())),
                  **{k:r[k] for k in ('sha256','bytes','records','stored_tokens','recomputed_tokens','sources')}} for r in audit['files']],
        'remaining_release_evidence':['reviewed final selection and quality evaluation',
            'per-source attribution and redistribution treatment','near-duplicate treatment and lineage',
            'final assembly integrity and manifest','dataset card matching final bytes'],
        'scope':'Draft inventory of existing retained files only; no publishable subset or final source precedence chosen. '
                'Conditional license states require their notices/attribution; an allowed state alone is not publication approval.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv)
    project=args.project_root.resolve()
    output=args.output_dir.resolve()
    if output==project or project.is_relative_to(output) or output.is_relative_to(project/'data'):
        parser.error('Keep review output separate from corpus data')
    config_path=project/'config/retained_baseline.json'
    policy_path=project/'config/source_licenses.yaml'
    config=json.loads(config_path.read_text())
    paths=[]
    for relative in config['inputs']:
        path=(project/relative).resolve()
        if not path.is_relative_to(project):
            raise ValueError('Baseline path escapes project')
        paths.append(path)
    output.mkdir(parents=True,exist_ok=True)
    with _directory_lock(output):
        if (output/'summary.json').exists():
            raise FileExistsError('Preserve completed preflight; choose a new output directory')
        audit_path=output/'integrity.json'
        status=integrity.main([*[str(p) for p in paths],'--workers',str(args.workers),'--output',str(audit_path)])
        if status not in (0,2):
            raise RuntimeError(f'Integrity scanner failed: {status}')
        audit=json.loads(audit_path.read_text())
        if audit['issues'].get('unreadable_or_invalid_file') or audit['issues'].get('file_changed_during_audit'):
            raise ValueError('Cannot build inventory from unreadable or changing files; inspect integrity.json')
        policy=yaml.safe_load(policy_path.read_text())
        result=draft_manifest(project,audit,policy,config)
        result['input_bindings']={'baseline_config_sha256':_sha256(config_path),
            'license_policy_sha256':_sha256(policy_path),'integrity_report_sha256':_sha256(audit_path),
            'code':{m.__name__:_sha256(Path(m.__file__)) for m in (integrity,licenses)},
            'runner_sha256':_sha256(Path(__file__))}
        _write_json(output/'manifest.draft.json',result)
        summary={'scan_complete':True,'release_ready':False,
            'records':result['records'],'recomputed_tokens':result['recomputed_tokens'],
            'matches_reference_inventory':result['matches_reference_inventory'],
            'integrity_passed':result['integrity_passed'],'integrity_issues':result['integrity_issues'],
            'license_state_totals':result['license_inventory']['state_totals'],
            'manifest_sha256':_sha256(output/'manifest.draft.json'),
            'scope':'Completed inventory, not release approval; review findings before assembly.'}
        _write_json(output/'summary.json',summary)
        print('BASELINE PREFLIGHT COMPLETE: '+json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
