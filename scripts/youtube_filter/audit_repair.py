"""Read-only audit of a completed repair attempt; produce a local review packet.

A successful audit can still report unresolved classification evidence. It never
changes source/repair outputs, supplies missing model citations, or selects data.
Only an already cached tokenizer is used; this command cannot load model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from . import repair_evidence as repair, rubric, rubric_scope, rubric_segments, score


def audit(pilot: Path, source: Path, repaired: Path):
    rows, tasks, originals, source_config, source_binding = repair.load_source(pilot, source)
    config, summary = repair.read(repaired/'run-config.json'), repair.read(repaired/'summary.json')
    if (config['source_binding'] != source_binding or config['original_config'] != source_config
            or config['version'] != repair.VERSION or config['dry_run'] or summary['dry_run']):
        raise ValueError('Repair does not bind to the original inference run')
    if (summary['model'] != source_config['model'] or summary['model_revision'] != source_config['model_revision']
            or summary['repair_version'] != repair.VERSION):
        raise ValueError('Repair summary model/version mismatch')
    for name, field in [('run-config.json','run_config_sha256'), ('requests.jsonl','requests_sha256'),
                        ('resolved_spans.jsonl','resolved_spans_sha256'), ('document_report.jsonl','document_report_sha256')]:
        if _sha256(repaired/name) != summary[field]:
            raise ValueError(f'Repair checksum mismatch: {name}')
    modules = [repair, score, rubric, rubric_scope, rubric_segments]
    if config['code'] != {Path(m.__file__).name: _sha256(Path(m.__file__)) for m in modules}:
        raise ValueError('Audit requires the recorded repair/parser implementations')
    tokenizer, snapshot = score.load_tokenizer(source_config['model'], source_config['model_revision'], True)
    if any(_sha256(snapshot/name) != value for name,value in source_config['tokenizer_files'].items()):
        raise ValueError('Tokenizer fingerprint mismatch')
    documents = {r['content_hash']:r for r in rows}
    for task in tasks:
        text = documents[task['content_hash']]['text']
        prompt, ids = rubric_scope.render(tokenizer, text, task['start'], task['end'], source_config['context_chars'])
        payload = {k:v for k,v in task.items() if k != 'task_sha256'}
        payload['prompt_token_ids'] = ids
        if (prompt != task['prompt'] or repair.digest(payload) != task['task_sha256']
                or task['response_schema'] != rubric_scope.response_schema(text[task['start']:task['end']])):
            raise ValueError('Original request does not reproduce from source text')
    selected = [t for t in tasks if originals[t['key']]['parse_status'] != 'ok']
    requests = [repair.make_request(t, documents[t['content_hash']], originals[t['key']], tokenizer,
        source_config['max_model_len'], config['max_output_tokens'], source_config['context_chars']) for t in selected]
    saved_requests = [json.loads(line) for line in (repaired/'requests.jsonl').read_text().splitlines()]
    if [{k:v for k,v in r.items() if k != 'prompt_token_ids'} for r in requests] != saved_requests:
        raise ValueError('Repair requests do not reproduce')
    if {p.stem for p in (repaired/'repairs').glob('*.json')} != {r['key'] for r in requests}:
        raise ValueError('Repair checkpoints do not cover the selected requests exactly once')
    states = originals.copy()
    raw_repairs, jobs = {}, Counter()
    for task, request in zip(selected, requests):
        key = task['key']
        provenance = {'original_decision_sha256': source_binding['decisions'][key],
            'model': source_config['model'], 'model_revision': source_config['model_revision'], 'repair_version': repair.VERSION}
        states[key] = repair.cached(repaired/'repairs'/f'{key}.json', request, originals[key],
            documents[task['content_hash']]['text'][task['start']:task['end']], task['start'], summary['run_config_sha256'], provenance)
        raw_repairs[key] = json.loads(states[key]['attempts'][-1]['raw_response'])
        for attempt in states[key]['attempts']:
            jobs[attempt.get('slurm_job_id') or 'not-recorded'] += 1
    merged = [json.loads(line) for line in (repaired/'resolved_spans.jsonl').read_text().splitlines()]
    if len(merged) != len(tasks):
        raise ValueError('Merged span count mismatch')
    unresolved, label_counts = [], Counter()
    for task, row in zip(tasks, merged):
        key, state = task['key'], states[task['key']]
        was_repaired = key in raw_repairs
        expected = {'key':key, 'content_hash':task['content_hash'], 'start':task['start'], 'end':task['end'],
            'original_decision_sha256':source_binding['decisions'][key],
            'origin': repair.VERSION if was_repaired else 'original_v3',
            'repair_decision_sha256':_sha256(repaired/'repairs'/f'{key}.json') if was_repaired else None,
            **{k:state.get(k) for k in ['parse_status','labels','evidence_offsets','error','repair_explanation']},
            'selection_decision':None}
        if row != expected:
            raise ValueError(f'Merged result does not reproduce: {key}')
        original_labels = json.loads(originals[key]['attempts'][-1]['raw_response'])
        if state['parse_status'] == 'ok':
            if {k:v for k,v in state['labels'].items() if k != 'evidence'} != {k:v for k,v in original_labels.items() if k != 'evidence'}:
                raise ValueError('Evidence repair changed an assessment field')
            label_counts[state['labels']['security_relevance']] += 1
        else:
            document = documents[task['content_hash']]
            unresolved.append({**row, 'original_assessment':original_labels, 'repair_response':raw_repairs.get(key),
                'focal_text':document['text'][task['start']:task['end']], 'sample_arms':document['sample_arms'],
                'source_locations':document['locations'],
                'review_status':'unresolved; no automatic keep/drop decision'})
    reports = repair.aggregate(rows, tasks, states)
    if reports != [json.loads(line) for line in (repaired/'document_report.jsonl').read_text().splitlines()]:
        raise ValueError('Document coverage report does not reproduce')
    errors = Counter(r['error'] for r in unresolved)
    expected_totals = {'source_spans':len(tasks), 'preserved_valid_spans':len(tasks)-len(selected),
        'repair_spans':len(selected), 'repaired_spans':len(selected)-len(unresolved), 'unresolved_spans':len(unresolved),
        'documents':len(rows), 'candidate_tokens':sum(r['content_length'] for r in rows),
        'validation_errors':dict(errors), 'complete':not unresolved}
    if any(summary[key] != value for key,value in expected_totals.items()):
        raise ValueError('Repair summary does not reproduce')
    result = {'audit_complete':True, 'evidence_coverage_complete':not unresolved,
        'classification_accuracy_established':False, 'audit_scope':'Artifact integrity and coverage only; no final selection or automatic resolution of missing evidence.',
        'repair_summary_sha256':_sha256(repaired/'summary.json'),
        **{k:v for k,v in expected_totals.items() if k != 'complete'},
        'spans_with_valid_evidence':len(tasks)-len(unresolved),
        'documents_with_complete_evidence':sum(r['coverage_complete'] for r in reports),
        'documents_requiring_review':sum(not r['coverage_complete'] for r in reports),
        'saved_repair_attempts_by_job':dict(jobs), 'valid_span_relevance_labels':dict(label_counts)}
    return result, unresolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir','source-dir','repair-dir','output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    for path in [args.input_dir,args.source_dir,args.repair_dir]:
        path = path.resolve()
        if path == output or path.is_relative_to(output) or output.is_relative_to(path):
            parser.error('Review output must be separate from all input directories')
    result, unresolved = audit(args.input_dir, args.source_dir, args.repair_dir)
    output.mkdir(parents=True, exist_ok=True)
    packet = output/'review_required.jsonl'
    packet.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in unresolved))
    result['review_packet_sha256'] = _sha256(packet)
    _write_json(output/'audit.json', result)
    print(json.dumps(result, indent=2))
    # Exit 0 certifies the read-only audit, not resolution or corpus eligibility.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
