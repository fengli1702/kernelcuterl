"""
Shared protocol utilities for evaluation pipeline modules.

Provides timing, logging, and result formatting functions.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def started_timer() -> float:
    """Start a performance counter for timing measurement."""
    return perf_counter()


def elapsed_ms(start: float) -> int:
    """Calculate elapsed time in milliseconds from a timer start."""
    return int((perf_counter() - start) * 1000)


def truncate_text(text: str | None, max_chars: int = 4000) -> str:
    """Truncate text to max_chars, appending truncation marker if needed."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>..."


def init_module_result(module_name: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Initialize a module result dictionary with standard structure."""
    return {
        "module": module_name,
        "ok": False,
        "status": "init",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "duration_ms": None,
        "input": deepcopy(request) if request else {},
        "metrics": {},
        "artifacts": {},
        "issues": [],
        "errors": [],
    }


def finalize_module_result(result: dict[str, Any], start: float, ok: bool, status: str) -> dict[str, Any]:
    """Finalize a module result with completion info."""
    result["ok"] = ok
    result["status"] = status
    result["finished_at"] = utc_now_iso()
    result["duration_ms"] = elapsed_ms(start)
    return result


def add_issue(
    result: dict[str, Any],
    issue_id: str,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Add an issue to module result."""
    issue = {
        "id": issue_id,
        "severity": severity,
        "message": message,
    }
    if details:
        issue["details"] = details
    result["issues"].append(issue)


def add_error(
    result: dict[str, Any],
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Add an error to module result."""
    error = {
        "type": error_type,
        "message": message,
    }
    if details:
        error["details"] = details
    result["errors"].append(error)
