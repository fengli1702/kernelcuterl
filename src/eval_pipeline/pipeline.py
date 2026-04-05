from __future__ import annotations

import re
from typing import Any

from src.eval_pipeline.compile_module import run_compile_module
from src.eval_pipeline.hack_check_module import run_hack_check_module
from src.eval_pipeline.protocol import elapsed_ms, started_timer, utc_now_iso
from src.eval_pipeline.timing_module import run_timing_module


DEFAULT_GATES = {
    "require_compile_ok": True,
    "require_timing_ok": False,
    "require_hack_clean": True,
}


def _is_generated_truncated(code: str | None, min_len: int) -> tuple[bool, str]:
    if not isinstance(code, str):
        return True, "generated_missing"

    s = code.strip()
    if not s:
        return True, "generated_empty"
    if len(s) < min_len:
        return True, "generated_too_short"

    trunc_markers = [
        "...<truncated>...",
        "<truncated>",
        "[truncated]",
        "TO_BE_CONTINUED",
    ]
    low = s.lower()
    if any(m.lower() in low for m in trunc_markers):
        return True, "generated_contains_trunc_marker"

    # Heuristic: if markdown fence appears odd number of times, content may be cut.
    fence_count = len(re.findall(r"```", s))
    if fence_count % 2 == 1:
        return True, "generated_unbalanced_markdown_fence"

    return False, "generated_ok"


def run_eval_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    """
    Unified evaluation pipeline (dict in, dict out).

    Request schema (top-level):
    - sample_id: str | int (optional)
    - reference_code: str (optional)
    - generated_code: str (optional)
    - generated_code_truncated: bool (optional)
    - use_reference_on_generated_truncation: bool (optional, default True)
    - generated_code_min_len: int (optional, default 48)
    - compile: dict (optional)
    - timing: dict (optional)
    - hack_check: dict (optional)
    - gates: dict (optional)

    Return schema:
    {
      "ok": bool,
      "status": "passed|failed",
      "sample_id": ...,
      "started_at": str,
      "finished_at": str,
      "duration_ms": int,
      "gates": {...},
      "gate_results": {...},
      "modules": {
        "hack_check": {...},
        "compile": {...},
        "timing": {...}
      },
      "summary": {...}
    }
    """
    start = started_timer()
    started_at = utc_now_iso()

    sample_id = request.get("sample_id")

    compile_req = dict(request.get("compile") or {})
    timing_req = dict(request.get("timing") or {})
    hack_req = dict(request.get("hack_check") or {})
    gates = {**DEFAULT_GATES, **dict(request.get("gates") or {})}

    reference_code = request.get("reference_code")
    generated_code = request.get("generated_code")
    generated_code_min_len = int(request.get("generated_code_min_len", 48))
    use_reference_on_generated_truncation = bool(request.get("use_reference_on_generated_truncation", True))
    explicit_truncated = request.get("generated_code_truncated")

    inferred_truncated, truncated_reason = _is_generated_truncated(generated_code, min_len=generated_code_min_len)
    is_generated_truncated = bool(explicit_truncated) if explicit_truncated is not None else inferred_truncated

    use_reference_as_generated = (
        is_generated_truncated
        and use_reference_on_generated_truncation
        and isinstance(reference_code, str)
        and bool(reference_code.strip())
    )

    effective_generated_code = reference_code if use_reference_as_generated else generated_code

    compile_req.setdefault("reference_code", reference_code)
    compile_req.setdefault("generated_code", generated_code)
    compile_req.setdefault("effective_generated_code", effective_generated_code)
    compile_req.setdefault("generated_code_truncated", is_generated_truncated)
    compile_req.setdefault("generated_code_truncation_reason", truncated_reason)
    if not compile_req.get("code") and isinstance(effective_generated_code, str):
        compile_req["code"] = effective_generated_code

    timing_req.setdefault("reference_code", reference_code)
    timing_req.setdefault("generated_code", generated_code)
    timing_req.setdefault("effective_generated_code", effective_generated_code)
    timing_req.setdefault("generated_code_truncated", is_generated_truncated)
    timing_req.setdefault("generated_code_truncation_reason", truncated_reason)

    # Keep timing callable path aligned with code fallback behavior.
    if (
        use_reference_as_generated
        and bool(timing_req.get("use_reference_fn_on_generated_truncation", True))
        and callable(timing_req.get("reference_fn"))
    ):
        timing_req["generated_fn"] = timing_req.get("reference_fn")
        if "reference_fn_args" in timing_req and "generated_fn_args" not in timing_req:
            timing_req["generated_fn_args"] = timing_req.get("reference_fn_args")
        if "reference_fn_kwargs" in timing_req and "generated_fn_kwargs" not in timing_req:
            timing_req["generated_fn_kwargs"] = timing_req.get("reference_fn_kwargs")
        if "reference_setup_fn" in timing_req and "generated_setup_fn" not in timing_req:
            timing_req["generated_setup_fn"] = timing_req.get("reference_setup_fn")

    if not hack_req.get("code") and isinstance(effective_generated_code, str):
        hack_req["code"] = effective_generated_code

    hack_result = run_hack_check_module(hack_req)
    compile_result = run_compile_module(compile_req) if compile_req else {
        "module": "compile",
        "ok": False,
        "status": "skipped",
        "issues": [{"id": "compile_missing", "severity": "medium", "message": "Compile request is missing"}],
        "errors": [],
        "metrics": {},
        "artifacts": {},
    }

    timing_allowed_without_compile = bool(timing_req.get("allow_without_compile", False))
    can_run_timing = bool(timing_req) and (compile_result.get("ok") or timing_allowed_without_compile)

    if can_run_timing:
        timing_result = run_timing_module(timing_req)
    else:
        timing_result = {
            "module": "timing",
            "ok": False,
            "status": "skipped",
            "issues": [
                {
                    "id": "timing_skipped_due_to_compile",
                    "severity": "medium",
                    "message": "Timing skipped because compile did not pass and allow_without_compile is false",
                }
            ],
            "errors": [],
            "metrics": {},
            "artifacts": {},
        }

    gate_results = {
        "compile_gate": (not gates["require_compile_ok"]) or bool(compile_result.get("ok")),
        "timing_gate": (not gates["require_timing_ok"]) or bool(timing_result.get("ok")),
        "hack_gate": (not gates["require_hack_clean"]) or bool(hack_result.get("ok")),
    }

    passed = all(gate_results.values())
    status = "passed" if passed else "failed"

    issues_total = (
        len(hack_result.get("issues", []))
        + len(compile_result.get("issues", []))
        + len(timing_result.get("issues", []))
    )
    errors_total = (
        len(hack_result.get("errors", []))
        + len(compile_result.get("errors", []))
        + len(timing_result.get("errors", []))
    )

    return {
        "ok": passed,
        "status": status,
        "sample_id": sample_id,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "duration_ms": elapsed_ms(start),
        "gates": gates,
        "gate_results": gate_results,
        "modules": {
            "hack_check": hack_result,
            "compile": compile_result,
            "timing": timing_result,
        },
        "summary": {
            "issues_total": issues_total,
            "errors_total": errors_total,
            "code_selection": {
                "generated_code_truncated": is_generated_truncated,
                "generated_code_truncation_reason": truncated_reason,
                "used_reference_as_generated": use_reference_as_generated,
                "used_reference_fn_as_generated": bool(
                    use_reference_as_generated
                    and bool(timing_req.get("use_reference_fn_on_generated_truncation", True))
                    and callable(timing_req.get("reference_fn"))
                ),
            },
            "module_status": {
                "hack_check": hack_result.get("status"),
                "compile": compile_result.get("status"),
                "timing": timing_result.get("status"),
            },
            "compile": {
                "return_code": (compile_result.get("metrics") or {}).get("return_code"),
                "error_category": (compile_result.get("metrics") or {}).get("error_category"),
                "warnings_count": (compile_result.get("metrics") or {}).get("warnings_count"),
                "errors_count": (compile_result.get("metrics") or {}).get("errors_count"),
            },
            "timing": {
                "mean_ms": (timing_result.get("metrics") or {}).get("mean_ms"),
                "p90_ms": (timing_result.get("metrics") or {}).get("p90_ms"),
                "success_rate": (timing_result.get("metrics") or {}).get("success_rate"),
                "performance_verdict": ((timing_result.get("artifacts") or {}).get("performance") or {}).get("verdict"),
                "speedup_vs_baseline": ((timing_result.get("artifacts") or {}).get("performance") or {}).get("speedup_vs_baseline"),
            },
            "hack_check": {
                "high": (hack_result.get("metrics") or {}).get("high"),
                "medium": (hack_result.get("metrics") or {}).get("medium"),
                "low": (hack_result.get("metrics") or {}).get("low"),
                "issue_by_category": (hack_result.get("metrics") or {}).get("issue_by_category"),
                "torch_reference_count": ((hack_result.get("artifacts") or {}).get("torch_usage") or {}).get("torch_reference_count"),
                "torch_import_count": ((hack_result.get("artifacts") or {}).get("torch_usage") or {}).get("torch_import_count"),
                "torch_symbol_total": ((hack_result.get("artifacts") or {}).get("torch_usage") or {}).get("torch_symbol_total"),
                "suspicious_torch_op_count": ((hack_result.get("artifacts") or {}).get("torch_usage") or {}).get("suspicious_torch_op_count"),
            },
        },
    }
