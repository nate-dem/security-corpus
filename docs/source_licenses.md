# Source-license release gate

The corpus does not have one blanket data license. Every release artifact must
retain its per-record `license` value and must be assembled only after
`scripts/release/audit_source_licenses.py` passes for the exact files being
uploaded. The machine-readable policy is `config/source_licenses.yaml`.

This is an engineering control, not legal advice. Stanford affiliation and a
PI's instruction to publish do not change third-party terms; they make it
appropriate to obtain institutional review or written permission for blocked
sources.

## Current source findings

| Source | Recorded term | Release state | Required action |
|---|---|---|---|
| NVD/CVE | CVE Terms of Use / NIST public data | Conditional | Include the CVE license/copyright notice and NIST attribution. |
| CISA KEV | CC0-1.0 | Conditional | Preserve provenance; do not imply endorsement. |
| MITRE ATT&CK/CWE/CAPEC | MITRE Terms of Use | Conditional | Reproduce the applicable copyright designation and license. |
| Sigma rules | DRL-1.1 | Conditional | Preserve rule-author attribution and the DRL notice. |
| Stack Exchange family | CC BY-SA by contribution date | Blocked for the recovered rows | Re-ingest with per-contribution author/license metadata or approve a reviewed attribution design. |
| Reddit | No redistribution license established | Blocked | Obtain written permission or an institutionally reviewed agreement. |
| flaws.cloud logs | No authoritative license stored | Blocked | Recover the original provenance and permission. |
| arXiv | Per-paper | Per license | Exclude the default arXiv license from uploaded full text; review NC/ND variants individually. |

Primary source references:

- [CVE Terms of Use](https://www.cve.org/legal/termsofuse)
- [CISA KEV license](https://www.cisa.gov/sites/default/files/licenses/kev/license.txt)
- [MITRE ATT&CK Terms of Use](https://attack.mitre.org/resources/terms-of-use/)
- [CWE Terms of Use](https://cwe.mitre.org/about/termsofuse.html)
- [CAPEC Terms of Use](https://capec.mitre.org/about/termsofuse.html)
- [Sigma DRL-1.1 repository notice](https://github.com/SigmaHQ/sigma/blob/master/LICENSE)
- [Stack Overflow contribution licenses](https://stackoverflow.com/help/licensing)
- [Reddit Public Content Policy](https://redditinc.com/policies/public-content-policy)
- [arXiv license guide](https://info.arxiv.org/help/license/index.html)
- [arXiv bulk-data reuse notice](https://info.arxiv.org/help/bulk_data_s3.html)

## Code license

The researcher must choose the repository code license before the public GitHub
release. Until a `LICENSE` file is added, the code is not presented as an
open-source grant.
