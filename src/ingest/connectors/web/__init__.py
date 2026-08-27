"""Connectors and helpers for general web corpora."""

from ingest.connectors.web.fineweb import (
    FINEWEB_LICENSE,
    FINEWEB_SOURCE_ID,
    DsirScorer,
    FineWebInputError,
    audit_fineweb_output,
    build_slurm_script,
    docs_from_input,
    fit_dsir_scorer,
    fineweb_record_text,
    normalize_fineweb_record,
    write_fineweb_records,
)

__all__ = [
    "FINEWEB_LICENSE",
    "FINEWEB_SOURCE_ID",
    "DsirScorer",
    "FineWebInputError",
    "audit_fineweb_output",
    "build_slurm_script",
    "docs_from_input",
    "fit_dsir_scorer",
    "fineweb_record_text",
    "normalize_fineweb_record",
    "write_fineweb_records",
]
