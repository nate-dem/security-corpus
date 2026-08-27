"""Structured Security Scope operation validation and rendering."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Sequence

from securityclip.paths import normalize_virtual_path
from securityclip.web.settings import WebSettings


ALLOWED_ROOTS = ("/papers", "/qa", "/vulns", "/rules", "/logs", "/knowledge", "/transcripts", "/web")
ALLOWED_TOOLS = {"search", "grep", "grep_from", "cat", "head", "ls"}
HANDLE_RE = re.compile(r"^s_[0-9a-fA-F]{8,}$")


@dataclass(frozen=True)
class ValidatedOperation:
    tool: str
    params: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"tool": self.tool, **self.params}


class OperationValidationError(ValueError):
    pass


def validate_operations(
    raw_operations: Any,
    settings: WebSettings,
    *,
    allowed_roots: Sequence[str] = ALLOWED_ROOTS,
) -> tuple[list[ValidatedOperation], list[str]]:
    if isinstance(raw_operations, dict) and "operations" in raw_operations:
        raw_operations = raw_operations["operations"]
    if not isinstance(raw_operations, list):
        raise OperationValidationError("planner output must be a JSON array of operations")
    if len(raw_operations) > settings.max_commands:
        raw_operations = raw_operations[: settings.max_commands]

    operations: list[ValidatedOperation] = []
    warnings: list[str] = []
    for idx, raw in enumerate(raw_operations, start=1):
        try:
            operation, op_warnings = validate_operation(raw, settings, allowed_roots=allowed_roots)
        except OperationValidationError as exc:
            raise OperationValidationError(f"operation {idx}: {exc}") from exc
        operations.append(operation)
        warnings.extend(op_warnings)
    return operations, warnings


def validate_operation(
    raw: Any,
    settings: WebSettings,
    *,
    allowed_roots: Sequence[str] = ALLOWED_ROOTS,
) -> tuple[ValidatedOperation, list[str]]:
    if isinstance(raw, str):
        raise OperationValidationError("raw shell command strings are not allowed")
    if not isinstance(raw, dict):
        raise OperationValidationError("operation must be a JSON object")
    tool = raw.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise OperationValidationError(f"unknown tool: {tool!r}")

    allowed_fields = _allowed_fields(tool)
    unknown = sorted(set(raw) - allowed_fields)
    if unknown:
        raise OperationValidationError(f"unsupported field(s): {', '.join(unknown)}")

    warnings: list[str] = []
    if tool == "search":
        query = _required_str(raw, "query")
        limit, clamped = _clamp_int(raw.get("limit", settings.max_results), 1, settings.max_limit)
        if clamped:
            warnings.append(f"search limit clamped to {limit}")
        return ValidatedOperation("search", {"query": query, "limit": limit}), warnings

    if tool == "grep":
        pattern = _required_str(raw, "pattern")
        _validate_regex(pattern)
        path = _validate_path(raw.get("path") or "/", allowed_roots)
        limit, clamped = _clamp_int(raw.get("limit", settings.max_results), 1, settings.max_limit)
        if clamped:
            warnings.append(f"grep limit clamped to {limit}")
        return ValidatedOperation(
            "grep",
            {
                "pattern": pattern,
                "path": path,
                "ignore_case": bool(raw.get("ignore_case", False)),
                "limit": limit,
            },
        ), warnings

    if tool == "grep_from":
        handle = _required_str(raw, "handle")
        if not HANDLE_RE.match(handle):
            raise OperationValidationError("grep_from handle must look like s_<hex>")
        pattern = _required_str(raw, "pattern")
        _validate_regex(pattern)
        limit, clamped = _clamp_int(raw.get("limit", settings.max_results), 1, settings.max_limit)
        if clamped:
            warnings.append(f"grep_from limit clamped to {limit}")
        return ValidatedOperation(
            "grep_from",
            {
                "handle": handle,
                "pattern": pattern,
                "ignore_case": bool(raw.get("ignore_case", False)),
                "limit": limit,
            },
        ), warnings

    if tool == "cat":
        path = _validate_path(_required_str(raw, "path"), allowed_roots)
        if path.endswith("/content.lines"):
            raise OperationValidationError("cat on content.lines is not allowed for planner operations; use head")
        return ValidatedOperation("cat", {"path": path}), warnings

    if tool == "head":
        path = _validate_path(_required_str(raw, "path"), allowed_roots)
        count, clamped = _clamp_int(raw.get("count", 50), 1, settings.max_head_count)
        if clamped:
            warnings.append(f"head count clamped to {count}")
        return ValidatedOperation("head", {"path": path, "count": count}), warnings

    if tool == "ls":
        path = _validate_path(str(raw.get("path") or "/"), allowed_roots)
        limit, clamped = _clamp_int(raw.get("limit", settings.max_results), 1, settings.max_limit)
        if clamped:
            warnings.append(f"ls limit clamped to {limit}")
        return ValidatedOperation("ls", {"path": path, "limit": limit}), warnings

    raise OperationValidationError(f"unhandled tool: {tool}")


def render_command(operation: ValidatedOperation) -> str:
    op = operation.params
    if operation.tool == "search":
        return f"security-scope search {shlex.quote(op['query'])} -n {op['limit']}"
    if operation.tool == "grep":
        bits = ["security-scope", "grep"]
        if op.get("ignore_case"):
            bits.append("-i")
        bits.extend([shlex.quote(op["pattern"]), shlex.quote(op["path"]), "--limit", str(op["limit"])])
        return " ".join(bits)
    if operation.tool == "grep_from":
        bits = ["security-scope", "grep", "--from", shlex.quote(op["handle"])]
        if op.get("ignore_case"):
            bits.append("-i")
        bits.extend([shlex.quote(op["pattern"]), "--limit", str(op["limit"])])
        return " ".join(bits)
    if operation.tool == "cat":
        return f"security-scope cat {shlex.quote(op['path'])}"
    if operation.tool == "head":
        return f"security-scope head -{op['count']} {shlex.quote(op['path'])}"
    if operation.tool == "ls":
        return f"security-scope ls {shlex.quote(op['path'])} --limit {op['limit']}"
    return f"security-scope {operation.tool} {json.dumps(op, sort_keys=True)}"


def _allowed_fields(tool: str) -> set[str]:
    return {
        "search": {"tool", "query", "limit"},
        "grep": {"tool", "pattern", "path", "ignore_case", "limit"},
        "grep_from": {"tool", "handle", "pattern", "ignore_case", "limit"},
        "cat": {"tool", "path"},
        "head": {"tool", "path", "count"},
        "ls": {"tool", "path", "limit"},
    }[tool]


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OperationValidationError(f"{key} must be a non-empty string")
    _reject_shellish(value)
    return value.strip()


def _reject_shellish(value: str) -> None:
    if any(token in value for token in ("\x00", "$(", "`", "&&", "||", ";", "|", ">", "<")):
        raise OperationValidationError("shell metacharacters are not allowed")


def _validate_regex(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise OperationValidationError(f"invalid regex: {exc}") from exc


def _validate_path(value: str, allowed_roots: Sequence[str] = ALLOWED_ROOTS) -> str:
    _reject_shellish(value)
    if ".." in value.split("/"):
        raise OperationValidationError("path traversal is not allowed")
    path = normalize_virtual_path(value)
    if path == "/":
        return path
    if not any(path == root or path.startswith(root + "/") for root in ALLOWED_ROOTS):
        raise OperationValidationError(f"path is outside allowed roots: {path}")
    if not any(path == root or path.startswith(root + "/") for root in allowed_roots):
        raise OperationValidationError(f"path is outside selected roots: {path}")
    return path


def _clamp_int(value: Any, minimum: int, maximum: int) -> tuple[int, bool]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    clamped = min(max(parsed, minimum), maximum)
    return clamped, clamped != parsed
