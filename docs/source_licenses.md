# Source-license metadata

CyberOpen combines materials from multiple sources. Each record contains a
`license` field, and source URLs and available attribution metadata are retained.
A recorded declaration or source-terms label is not a single license for the
whole corpus.

The source families include per-paper arXiv licenses, contribution-specific
Stack Exchange terms, Sigma detection-rule terms, public-agency and MITRE source
terms, platform terms, and upstream web/transcript dataset declarations. For
aggregated datasets, a collection-level declaration does not establish the
rights of every underlying document. The label `Public` means publicly
available; it does not mean public domain.

`config/source_licenses.yaml` supplies the source-policy checks used by the
release-audit tooling. `scripts/release/rights_sources.json` records source-term
references. The tools under `scripts/release/` retain the distinction between
source declarations, record-level metadata, and redistribution decisions.
