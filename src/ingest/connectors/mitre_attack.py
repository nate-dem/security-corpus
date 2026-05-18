"""Compatibility alias for the moved MITRE ATT&CK connector."""

import sys

from ingest.connectors.knowledge import mitre_attack as _impl

sys.modules[__name__] = _impl

