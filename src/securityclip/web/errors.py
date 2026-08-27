"""Categorized error collection for Security Scope web runs."""

from __future__ import annotations


ERROR_CODES = (
    "missing_index",
    "missing_api_key",
    "router_failed",
    "planner_failed",
    "validation_failed",
    "repair_failed",
    "operation_timeout",
    "operation_failed",
    "no_results",
    "synthesis_failed",
)

DEFAULT_HINTS: dict[str, str] = {
    "missing_index": "Set SECURITYCLIP_INDEX to a directory containing securityclip.sqlite and restart Security Scope (`security-scope-web`).",
    "missing_api_key": "Set OPENAI_API_KEY (and OPENAI_BASE_URL if your provider needs one) and restart Security Scope (`security-scope-web`).",
    "router_failed": "Retry the query; if it persists, check the router model name and API connectivity.",
    "planner_failed": "Retry the query; if it persists, check the planner model name and API connectivity.",
    "validation_failed": "The planner emitted an invalid operation; it was adjusted or replaced with a fallback search.",
    "repair_failed": "The planner could not repair its invalid output; a fallback search was used.",
    "operation_timeout": "Increase SECURITYCLIP_COMMAND_TIMEOUT or narrow the operation scope.",
    "operation_failed": "Check the command trace for the failing operation.",
    "no_results": "Try broader terms or remove source filters.",
    "synthesis_failed": "The answer shown is a deterministic fallback; retry for a model-written report.",
}


class ErrorCollector:
    """Collects run errors as both legacy strings and categorized detail dicts.

    ``messages`` preserves the historical ``errors: list[str]`` response field
    (and the ``done``/``done_with_errors`` status semantics keyed on it);
    ``details`` adds ``{code, message, hint}`` objects for the UI. A detail
    added with ``detail_only=True`` is informational and does not flip the run
    status.
    """

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._details: list[dict[str, str]] = []

    def add(self, code: str, message: str, hint: str = "", *, detail_only: bool = False) -> None:
        if code not in ERROR_CODES:
            code = "operation_failed"
        if not detail_only:
            self._messages.append(message)
        self._details.append({"code": code, "message": message, "hint": hint or DEFAULT_HINTS.get(code, "")})

    @property
    def messages(self) -> list[str]:
        return list(self._messages)

    @property
    def details(self) -> list[dict[str, str]]:
        return [dict(detail) for detail in self._details]


def classify_operation_error(error: str) -> str:
    lowered = error.lower()
    if "timed out" in lowered:
        return "operation_timeout"
    if "index not found" in lowered:
        return "missing_index"
    return "operation_failed"
