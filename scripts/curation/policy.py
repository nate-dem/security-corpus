"""Conservative routing under the user's delegated quality-first review.

Eligible means suitable for candidate assembly, not cleared for publication.
No input text is removed by this module. Uncertain or invalid outputs stay open.
"""
from .rubric import QUALITY_CONCERNS
from scripts.youtube_filter.rubric import LABELS

VERSION='conservative-quality-v1'


def route(labels, parse_status='ok'):
    if parse_status!='ok' or not isinstance(labels,dict):
        return 'review_required'
    if any(labels.get(k) not in values for k,values in LABELS.items()):
        return 'review_required'
    concerns=labels.get('quality_concerns')
    if not isinstance(concerns,list):
        return 'review_required'
    kinds=[c.get('kind') if isinstance(c,dict) else c for c in concerns]
    if any(k not in QUALITY_CONCERNS for k in kinds):
        return 'review_required'
    if any(labels[k]=='uncertain' for k in LABELS):
        return 'review_required'
    if labels['security_relevance'] in {'absent','incidental'}:
        return 'exclude'
    if labels['text_language']=='other' or labels['text_usability']=='unusable':
        return 'exclude'
    if labels['technical_substance'] in {'limited','none'}:
        return 'exclude'
    if kinds or labels['mixed_content']!='no' or labels['text_language']!='english' or labels['text_usability']!='usable':
        return 'review_required'
    return 'eligible'
