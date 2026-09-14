"""Create a blinded local development review from already transferred artifacts.

No model predictions are shown or supplied as human labels. These examples have
already contributed to development and cannot establish held-out accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ingest.utils import compute_content_hash, compute_token_count
from scripts.web_corpora.profile import features, text_field
from scripts.youtube.profile import _sha256
from scripts.youtube_filter.rubric import LABELS


def read_rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def web_cases(reports, per_source, seed):
    cases, bindings = [], {}
    for source, version in [('redsage-cfw','profile-v1'),('primus-fineweb','profile-v2')]:
        root = reports / 'web-corpora' / source / version
        summary = json.loads((root/'summary.json').read_text())
        path = root/'inspection_sample.jsonl'
        if summary.get('complete') is not True or _sha256(path) != summary['inspection_sample']['sha256']:
            raise ValueError(f'Invalid sample provenance: {path}')
        bindings[str(path)] = _sha256(path)
        population = []
        seen = set()
        for row in read_rows(path):
            key = (row['source_shard'],row['source_row'])
            if key in seen or row['source'] != source or row['dataset_revision'] != summary['revision']:
                raise ValueError('Duplicate or inconsistent sample lineage')
            seen.add(key)
            measured = features(row['raw_record'],source,*key)
            if measured != row['features']:
                raise ValueError('Web sample does not reproduce from its source text')
            if measured['text_state']=='valid':
                population.append(row)
        ordered = sorted(population, key=lambda r: hashlib.sha256(
            f"{seed}:{source}:{r['source_shard']}:{r['source_row']}".encode()).hexdigest())
        for row in ordered[:per_source]:
            text = row['raw_record'][text_field(source)]
            case_id = f"{source}:{row['source_shard']}:{row['source_row']}"
            cases.append({'case_id':case_id,'source':source,'text':text,
                'text_sha256':compute_content_hash(text),'content_length':compute_token_count(text),
                'unit':'complete_document','purpose':'development_only',
                'lineage':{k:row[k] for k in ('dataset_revision','source_shard','source_row','selection_probability')},
                'selection_from_saved_valid_samples':min(per_source,len(population))/len(population),
                'sampling_note':'Deterministic subsample of saved per-shard diagnostic samples; not an independent holdout or unweighted corpus sample.'})
    return cases, bindings


def youtube_cases(reports):
    root = reports/'youtube'
    packet = root/'repair-review-v1/review_required.jsonl'
    audit = json.loads((packet.parent/'audit.json').read_text())
    if not audit['audit_complete'] or _sha256(packet) != audit['review_packet_sha256']:
        raise ValueError('Invalid repair review provenance')
    cases=[]
    for row in read_rows(packet):
        text=row['focal_text']
        cases.append({'case_id':row['key'],'source':'youtube-commons','text':text,
            'text_sha256':compute_content_hash(text),'content_length':compute_token_count(text),
            'unit':'focal_span','purpose':'development_only',
            'lineage':{k:row[k] for k in ('content_hash','start','end','source_locations')},
            'sampling_note':'All unresolved evidence cases, deliberately enriched; no prevalence estimate.'})
    return cases,{str(packet):_sha256(packet)}


def validate_labels(packet, annotations):
    digest=hashlib.sha256(json.dumps({k:v for k,v in packet.items() if k!='packet_sha256'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if packet.get('packet_sha256')!=digest:
        raise ValueError('Review packet contents changed')
    if annotations.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('Annotations belong to a different review packet')
    by_id={c['case_id']:c for c in packet['cases']}
    seen=set()
    completed=[]
    for row in annotations['annotations']:
        key=row['case_id']
        if key not in by_id or key in seen or row.get('text_sha256') != by_id[key]['text_sha256']:
            raise ValueError('Unknown, repeated, or changed annotation text')
        seen.add(key)
        if row.get('status') != 'reviewed':
            continue
        if set(row.get('labels',{})) != set(LABELS) or any(row['labels'][k] not in values for k,values in LABELS.items()):
            raise ValueError('Invalid or incomplete human labels')
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('reviewer','notes','reviewed_at')):
            raise ValueError('Reviewed annotations need reviewer, notes, and timestamp')
        completed.append(row)
    return {'reviewed':len(completed),'total_cases':len(by_id),'remaining':len(by_id)-len(completed),
        'held_out_accuracy_established':False,'selection_policy_approved':False,
        'source_counts':dict(Counter(by_id[r['case_id']]['source'] for r in completed))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    # RESEARCHER: review workload, not an eligibility threshold or power calculation.
    parser.add_argument('--web-per-source',type=int,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args(argv)
    if args.web_per_source < 1:
        parser.error('web-per-source must be positive')
    output=args.output_dir.resolve()
    reports=args.reports_dir.resolve()
    if output==reports or reports.is_relative_to(output):
        parser.error('Output must not replace the reports root')
    if output.exists():
        raise FileExistsError('Preserve existing review packets and labels; choose a new output directory')
    cases,bindings=web_cases(reports,args.web_per_source,args.seed)
    youtube,more=youtube_cases(reports)
    cases.extend(youtube)
    bindings.update(more)
    packet={'version':1,'purpose':'development_only','labels':LABELS,'cases':cases,'source_bindings':bindings,
        'web_per_source':args.web_per_source,'seed':args.seed,'selection_policy_approved':False}
    digest=hashlib.sha256(json.dumps(packet,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    packet['packet_sha256']=digest
    output.mkdir(parents=True)
    (output/'packet.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
    template=Path(__file__).with_name('review.html').read_text()
    # Source text never enters innerHTML. Escape script-closing tokens as well.
    embedded=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e')
    (output/'review.html').write_text(template.replace('/*PACKET_JSON*/',embedded))
    print(json.dumps({'cases':len(cases),'sources':dict(Counter(c['source'] for c in cases)),
        'packet_sha256':digest,'review_page':str(output/'review.html'),'held_out':False},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
