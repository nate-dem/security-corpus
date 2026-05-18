"""Compatibility alias for the moved MITRE CWE connector."""

import sys

from ingest.connectors.knowledge import mitre_cwe as _impl

sys.modules[__name__] = _impl

