"""
Hack checking module for code quality analysis.

Detects suspicious patterns including:
- Fake optimizations (direct passthrough)
- Backend signal verification (CUDA, Triton, HIP)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .protocol import (
    add_error,
    finalize_module_result,
    init_module_result,
    started_timer,
)


RULES: list[dict[str, str]] = [
    {
        "id": "fabricated_output",
        "category": "fake_optimization",
        "message": "Detected fabricated tensor output pattern",
        "pattern": r"return\s+torch\.(zeros_like|ones_like|rand_like|full_like|empty_like)\(",
    },
    {
        "id": "passthrough_return",
        "category": "fake_optimization",
        "message": "Detected direct passthrough return",
        "pattern": r"return\s+(x|input|inputs\[\d+\])\s*(#.*)?$",
    },
]


TORCH_OP_RULES = [
    r"\btorch\.matmul\(",
    r"\btorch\.mm\(",
    r"\btorch\.bmm\(",
    r"\btorch\.einsum\(",
    r"\btorch\.nn\.functional\.",
    r"\bF\.\w+\(",
]


BACKEND_SIGNAL_PATTERNS: dict[str, list[str]] = {
    "cuda": [
        r"__global__",
        r"torch\.utils\.cpp_extension",
        r"load_inline\(",
        r"torch\.ops\.",
        r"cupy\.RawKernel",
        r"nvcc",
    ],
    "triton": [
        r"@triton\.jit",
        r"triton\.jit",
        r"triton\.language",
        r"tl\.",
    ],
    "hip": [
        r"hip",
        r"__global__",
        r"torch\.utils\.cpp_extension",
        r"load_inline\(",
    ],
}


def _line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _snippet(text: str, start: int, end: int, radius: int = 100) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", "\\n")


def _collect_matches(payload: str, rules: list[dict[str, str]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for rule in rules:
        pattern = re.compile(rule["pattern"], flags=re.IGNORECASE | re.MULTILINE)
        for m in pattern.finditer(payload):
            issues.append(
                {
                    "id": rule["id"],
                    "category": rule["category"],
                    "message": rule["message"],
                    "line": _line_of_offset(payload, m.start()),
                    "span": [m.start(), m.end()],
                    "snippet": _snippet(payload, m.start(), m.end()),
                }
            )
    return issues


def _maybe_add_precision_issue(payload: str, precision_target: str | None, issues: list[dict[str, Any]]) -> None:
    if not precision_target:
        return
    pt = precision_target.lower().strip()
    if pt != "fp32":
        return

    patterns = [r"float16", r"bfloat16", r"\.half\(", r"\.to\(\s*torch\.(float16|bfloat16)"]
    for pat in patterns:
        p = re.compile(pat, flags=re.IGNORECASE)
        m = p.search(payload)
        if not m:
            continue
        issues.append(
            {
                "id": "precision_downgrade",
                "category": "fake_optimization",
                "message": "Detected potential precision downgrade under fp32 target",
                "line": _line_of_offset(payload, m.start()),
                "span": [m.start(), m.end()],
                "snippet": _snippet(payload, m.start(), m.end()),
            }
        )
        return


def _torch_usage_metrics(payload: str) -> dict[str, Any]:
    torch_refs = len(re.findall(r"\btorch\.", payload))
    torch_imports = len(re.findall(r"\bimport\s+torch\b|\bfrom\s+torch\b", payload))
    torch_symbol_total = torch_refs + torch_imports
    suspicious_hits: list[dict[str, Any]] = []

    for pat in TORCH_OP_RULES:
        p = re.compile(pat, flags=re.IGNORECASE)
        for m in p.finditer(payload):
            suspicious_hits.append(
                {
                    "pattern": pat,
                    "line": _line_of_offset(payload, m.start()),
                    "snippet": _snippet(payload, m.start(), m.end()),
                }
            )

    return {
        "torch_reference_count": torch_refs,
        "torch_import_count": torch_imports,
        "torch_symbol_total": torch_symbol_total,
        "suspicious_torch_op_hits": suspicious_hits,
        "suspicious_torch_op_count": len(suspicious_hits),
    }


def _custom_signal_metrics(payload: str, expected_backend: str | None, extra_patterns: list[str] | None) -> dict[str, Any]:
    backend = (expected_backend or "").lower().strip()
    pats = list(BACKEND_SIGNAL_PATTERNS.get(backend, []))
    if extra_patterns:
        pats.extend([x for x in extra_patterns if isinstance(x, str) and x.strip()])

    hits: list[dict[str, Any]] = []
    for pat in pats:
        p = re.compile(pat, flags=re.IGNORECASE | re.MULTILINE)
        for m in p.finditer(payload):
            hits.append(
                {
                    "pattern": pat,
                    "line": _line_of_offset(payload, m.start()),
                    "snippet": _snippet(payload, m.start(), m.end()),
                }
            )

    return {
        "expected_backend": backend or None,
        "custom_signal_patterns": pats,
        "custom_signal_hits": hits,
        "custom_signal_hit_count": len(hits),
    }


def run_hack_check_module(request: dict[str, Any]) -> dict[str, Any]:
    """
    Hack-check module interface (dict in, dict out).

    Request keys:
    - code: str (optional)
    - response_text: str (optional)
    - strict: bool (optional, default True)
    - torch_usage_policy: allow|warn|forbid (optional, default warn)
    - precision_target: fp32|fp16|bf16 (optional)
    - expected_backend: cuda|triton|hip (optional)
    - expected_custom_signals: list[str regex] (optional)
    - min_custom_signal_hits: int (optional, default 1)
    """
    start = started_timer()
    result = init_module_result("hack_check", request)

    try:
        strict = bool(request.get("strict", True))
        torch_usage_policy = str(request.get("torch_usage_policy", "warn")).lower().strip()
        precision_target = request.get("precision_target")
        expected_backend = request.get("expected_backend")
        expected_custom_signals = request.get("expected_custom_signals") or []
        min_custom_signal_hits = int(request.get("min_custom_signal_hits", 1))

        code = request.get("code")
        response_text = request.get("response_text")

        payload = ""
        if isinstance(code, str) and code.strip():
            payload = code
        elif isinstance(response_text, str) and response_text.strip():
            payload = response_text
        else:
            result["metrics"] = {
                "checked_chars": 0,
                "issue_total": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }
            result["artifacts"] = {
                "torch_usage": {},
                "custom_signal": {},
            }
            return finalize_module_result(result, start, ok=True, status="ok")

        issues = _collect_matches(payload, RULES)
        _maybe_add_precision_issue(payload, precision_target, issues)

        torch_metrics = _torch_usage_metrics(payload)
        signal_metrics = _custom_signal_metrics(payload, expected_backend, expected_custom_signals)

        if torch_usage_policy == "forbid" and torch_metrics["torch_symbol_total"] > 0:
            issues.append(
                {
                    "id": "torch_usage_forbidden",
                    "category": "torch_usage",
                    "message": "Torch usage is forbidden by policy",
                    "line": 1,
                    "span": [0, 0],
                    "snippet": "torch.* references detected",
                }
            )
        elif torch_usage_policy == "warn" and torch_metrics["torch_symbol_total"] > 0:
            issues.append(
                {
                    "id": "torch_usage_warning",
                    "category": "torch_usage",
                    "message": "Torch usage detected; verify this is not fallback",
                    "line": 1,
                    "span": [0, 0],
                    "snippet": "torch.* references detected",
                }
            )

        if torch_metrics["suspicious_torch_op_count"] > 0:
            issues.append(
                {
                    "id": "suspicious_torch_ops",
                    "category": "torch_usage",
                    "message": "Detected suspicious torch operator usage",
                    "line": torch_metrics["suspicious_torch_op_hits"][0]["line"],
                    "span": [0, 0],
                    "snippet": torch_metrics["suspicious_torch_op_hits"][0]["snippet"],
                    "details": {"count": torch_metrics["suspicious_torch_op_count"]},
                }
            )

        backend_norm = (expected_backend or "").lower().strip()
        if backend_norm in {"cuda", "triton", "hip"} and signal_metrics["custom_signal_hit_count"] < max(0, min_custom_signal_hits):
            issues.append(
                {
                    "id": "missing_custom_kernel_signal",
                    "category": "fake_optimization",
                    "message": "No enough custom kernel signals found",
                    "line": 1,
                    "span": [0, 0],
                    "snippet": "No expected backend signals were detected",
                    "details": {
                        "expected_backend": backend_norm,
                        "required_min_hits": min_custom_signal_hits,
                        "actual_hits": signal_metrics["custom_signal_hit_count"],
                    },
                }
            )

        result["issues"] = issues

        has_issue = len(issues) > 0

        by_category = Counter(x.get("category", "unknown") for x in issues)

        result["metrics"] = {
            "checked_chars": len(payload),
            "issue_total": len(issues),
            "has_issue": has_issue,
            "strict": strict,
            "torch_usage_policy": torch_usage_policy,
            "precision_target": precision_target,
            "issue_by_category": dict(by_category),
        }
        result["artifacts"] = {
            "torch_usage": torch_metrics,
            "custom_signal": signal_metrics,
        }

        if has_issue:
            return finalize_module_result(result, start, ok=False, status="failed")
        return finalize_module_result(result, start, ok=True, status="ok")

    except Exception as exc:  # pragma: no cover - defensive
        add_error(result, "unexpected_exception", str(exc))
        return finalize_module_result(result, start, ok=False, status="error")
