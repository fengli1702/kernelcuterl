"""
Kernel evaluation module.

Evaluates generated kernel code against reference implementations
using the eval_pipeline for comprehensive assessment.
"""

from typing import Any, Dict, Optional

from .pipeline import run_eval_pipeline


def eval(
    generated_code: Optional[str],
    reference_code: Optional[str],
    language: str = "cuda",
    use_compile: bool = True,
    use_timing: bool = False,
    use_correctness: bool = True,
    use_hack_check: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate generated kernel code against reference.

    Uses the eval_pipeline to assess:
    - Compilation success
    - Correctness (output comparison)
    - Code quality (hack checks)
    - Performance (timing) - optional

    Args:
        generated_code: Generated kernel code to evaluate
        reference_code: Reference/baseline kernel code
        language: Language type ("cuda", "python", "cpp")
        use_compile: Enable compilation check (default: True)
        use_timing: Enable timing/performance measurement (default: False)
        use_correctness: Enable correctness checking (default: True)
        use_hack_check: Enable code quality checks (default: True)

    Returns:
        Dict with evaluation results including:
        - correct: bool - Whether outputs match reference
        - compile_ok: bool - Compilation status
        - correctness_ok: bool - Correctness check status
        - hack_clean: bool - Code quality status
        - error: str - Error message if failed
        - timing_stats: dict - Performance metrics (if timing enabled)
        - pipeline_result: dict - Full pipeline result

    Example:
        >>> generated = "def kernel(x): return x * 2"
        >>> reference = "def kernel(x): return x * 2"
        >>> result = eval(generated, reference)
        >>> print(result.get("correct"))
        True
    """
    if not generated_code or not reference_code:
        return {
            "correct": False,
            "error": "missing_code",
            "compile_ok": False,
            "correctness_ok": False,
            "hack_clean": False,
        }

    try:
        # Build pipeline request
        pipeline_request = {
            "reference_code": reference_code,
            "generated_code": generated_code,
            "compile": {"language": language} if use_compile else None,
            "correctness": {} if use_correctness else None,
            "timing": {} if use_timing else None,
            "hack_check": {} if use_hack_check else None,
            "gates": {
                "require_compile_ok": use_compile,
                "require_correctness_ok": use_correctness,
                "require_timing_ok": False,
                "require_hack_clean": use_hack_check,
            },
        }

        # Remove None values
        pipeline_request = {k: v for k, v in pipeline_request.items() if v is not None}

        # Run evaluation pipeline
        pipeline_result = run_eval_pipeline(pipeline_request)

        # Extract key metrics
        compile_ok = pipeline_result.get("gate_results", {}).get("compile_gate", False)
        correctness_ok = pipeline_result.get("gate_results", {}).get("correctness_gate", False)
        hack_clean = pipeline_result.get("gate_results", {}).get("hack_gate", False)
        timing_ok = pipeline_result.get("gate_results", {}).get("timing_gate", False)

        # Get correctness result
        correctness_metrics = pipeline_result.get("summary", {}).get("correctness", {})
        correct = correctness_metrics.get("correct", False)

        # Determine overall correctness
        overall_ok = pipeline_result.get("ok", False)

        # Extract timing stats if available
        timing_summary = pipeline_result.get("summary", {}).get("timing", {})

        result = {
            "correct": correct,
            "compile_ok": compile_ok,
            "correctness_ok": correctness_ok,
            "hack_clean": hack_clean,
            "timing_ok": timing_ok,
            "pipeline_ok": overall_ok,
            "pipeline_result": pipeline_result,
        }

        # Add timing metrics if available
        if timing_summary and timing_summary.get("mean_ms"):
            result["timing_stats"] = {
                "mean_ms": timing_summary.get("mean_ms"),
                "median_ms": timing_summary.get("median_ms"),
                "p90_ms": timing_summary.get("p90_ms"),
                "speedup_vs_baseline": timing_summary.get("speedup_vs_baseline"),
                "performance_verdict": timing_summary.get("performance_verdict"),
            }

        # Add compile error details if compilation failed
        if not compile_ok:
            compile_metrics = pipeline_result.get("summary", {}).get("compile", {})
            result["compile_error_category"] = compile_metrics.get("error_category")
            result["compile_errors"] = compile_metrics.get("errors_count", 0)

        # Add correctness details
        if not correctness_ok:
            result["correctness_error_type"] = correctness_metrics.get("error_type")
            result["correctness_max_diff"] = correctness_metrics.get("max_diff")
            result["correctness_mean_diff"] = correctness_metrics.get("mean_diff")

        # Add hack check details if code quality issues found
        if not hack_clean:
            hack_metrics = pipeline_result.get("summary", {}).get("hack_check", {})
            result["hack_issues"] = hack_metrics.get("issue_total", 0)
            result["hack_issue_categories"] = hack_metrics.get("issue_by_category", {})

        return result

    except Exception as exc:
        return {
            "correct": False,
            "error": f"evaluation_failed: {str(exc)}",
            "compile_ok": False,
            "hack_clean": False,
            "traceback": str(exc),
        }


# Aliases for backward compatibility
evaluate = eval
evaluate_kernel = eval
