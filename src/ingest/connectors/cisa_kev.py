"""Compatibility alias for the moved CISA KEV connector."""

import sys

from ingest.connectors.vulnerability import cisa_kev as _impl

sys.modules[__name__] = _impl

