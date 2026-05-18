"""Compatibility alias for the moved NVD connector."""

import sys

from ingest.connectors.vulnerability import nvd as _impl

sys.modules[__name__] = _impl

