#!/usr/bin/env python3
"""
single_turn_rollout.py

Performs a single-turn rollout + evaluation for a given JSONL dataset.

Core behavior (based on --save-path):
1. If save-path does not exist: start rollout from scratch (with optional inline eval).
2. If save-path exists but is incomplete (record count < data total): resume rollout from checkpoint.
3. If save-path has a complete rollout but evaluation is incomplete: load existing results and re-run detector + eval.
4. If save-path exists and both rollout + evaluation are complete: read directly and print summary statistics.

File conventions (based on save-path):
- {save_path}                    Final complete result (rollout text + eval)
- {save_path}.checkpoint.jsonl   Temporary checkpoint for resume (auto-removed after completion)
- {save_path}.summary.json       Aggregated statistics
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import httpx

try:
    from tqdm.asyncio import tqdm as tqdm_async
except Exception:  # pragma: no cover
    tqdm_async = None  # type: ignore

try:
    from tqdm import tqdm as tqdm_sync
except Exception:  # pragma: no cover
    tqdm_sync = None  # type: ignore


CLOSE_TAG_RE = re.compile(r"(?is)</think>|</thinking>|</Think>|</Thought>|</THINK>|</THINKING>|</thinking>")
MODEL_CLASS_RE = re.compile(r"(?m)^\s*class\s+Model(?:New)?\b")
FENCED_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+#-]*)?\n?(.*?)\n?```", re.DOTALL)


# ---------------------------------------------------------------------------
# Try importing kernelrl eval tools; warn gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    from kernelrl.eval.detector import detector
    from kernelrl.eval.eval import eval as kernel_eval
except Exception as _import_exc:  # pragma: no cover
    detector = None  # type: ignore
    kernel_eval = None  # type: ignore
    _IMPORT_ERROR_MSG = (
        f"[WARN] Failed to import kernelrl.eval.detector / kernelrl.eval.eval: {_import_exc}\n"
        "The script will continue, but extraction and evaluation steps will be marked as failed."
    )
    print(_IMPORT_ERROR_MSG, file=sys.stderr)

# Try importing eval_pipeline; warn gracefully if unavailable
try:
    from kernelrl.eval.pipeline import run_eval_pipeline
except Exception as _pipeline_import_exc:  # pragma: no cover
    run_eval_pipeline = None  # type: ignore
    _PIPELINE_IMPORT_MSG = (
        f"[WARN] Failed to import kernelrl.eval.pipeline: {_pipeline_import_exc}\n"
        "The script will continue, but --use-pipeline flag will be unavailable."
    )
    print(_PIPELINE_IMPORT_MSG, file=sys.stderr)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    text: str
    kernel_code: Optional[str]
    eval_result: Optional[Dict[str, Any]]
    error: Optional[str] = None
    turns: Optional[List[Dict[str, Any]]] = None


@dataclass
class ItemResult:
    idx: int
    sample_id: Optional[int]
    prompt: str
    reference: str
    level: Optional[str]
    difficulty: Optional[int]
    file_path: Optional[str]
    samples: List[SampleResult]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file (one JSON object per line) or a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # Detect JSON array format (e.g. some kernelbench files are JSON arrays)
    if content.startswith("["):
        return json.loads(content)
    # Standard JSONL: one JSON object per line
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_jsonl(path: str, items: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def item_result_to_dict(r: ItemResult) -> Dict[str, Any]:
    return {
        "idx": r.idx,
        "sample_id": r.sample_id,
        "file_path": r.file_path,
        "level": r.level,
        "difficulty": r.difficulty,
        "prompt": r.prompt,
        "reference": r.reference,
        "samples": [
            {
                "text": s.text,
                "kernel_code": s.kernel_code,
                "eval_result": s.eval_result,
                "error": s.error,
                "turns": s.turns or [],
            }
            for s in r.samples
        ],
        "summary": r.summary,
    }


def dict_to_item_result(d: Dict[str, Any]) -> ItemResult:
    samples = [
        SampleResult(
            text=s.get("text", ""),
            kernel_code=s.get("kernel_code"),
            eval_result=s.get("eval_result"),
            error=s.get("error"),
            turns=s.get("turns") or [],
        )
        for s in d.get("samples", [])
    ]
    return ItemResult(
        idx=d.get("idx", 0),
        sample_id=d.get("sample_id"),
        prompt=d.get("prompt", ""),
        reference=d.get("reference", ""),
        level=d.get("level"),
        difficulty=d.get("difficulty"),
        file_path=d.get("file_path"),
        samples=samples,
        summary=d.get("summary", {}),
    )


def _extract_after_last_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = list(CLOSE_TAG_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end():]


def _has_model_class(code: str) -> bool:
    return bool(code and MODEL_CLASS_RE.search(code))


def _compile_status(code: str) -> tuple[bool, Optional[int]]:
    if not code:
        return False, None
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except SyntaxError as exc:
        return False, exc.lineno
    except Exception:
        return False, None


def _trim_to_compilable_prefix(code: str) -> str:
    if not code:
        return ""
    lines = code.strip("\n").splitlines()
    while lines:
        cand = "\n".join(lines).strip("\n")
        if not cand:
            return ""
        ok, err_lineno = _compile_status(cand)
        if ok:
            return cand
        if isinstance(err_lineno, int) and 1 <= err_lineno <= len(lines):
            next_len = err_lineno - 1
            if next_len < len(lines):
                lines = lines[:next_len]
                continue
        lines = lines[:-1]
    return ""


def _candidate_score(code: str) -> tuple[int, int, int, int, int, int]:
    ok, _ = _compile_status(code)
    return (
        1 if _has_model_class(code) else 0,
        1 if "def forward" in code else 0,
        1 if "def get_inputs" in code else 0,
        1 if "def get_init_inputs" in code else 0,
        1 if ok else 0,
        len(code),
    )


def _best_detector_candidate(cand: Any) -> tuple[str, str]:
    if cand is None:
        return "", "none"
    if isinstance(cand, str):
        text = cand.strip()
        return (text, "detector_single") if text else ("", "none")
    if isinstance(cand, (list, tuple)):
        blocks = [str(c).strip() for c in cand if str(c).strip()]
        if not blocks:
            return "", "none"
        merged: List[str] = []
        for i in range(len(blocks) - 1):
            merged.append("\n\n".join(blocks[i : i + 2]).strip())
        for i in range(len(blocks) - 2):
            merged.append("\n\n".join(blocks[i : i + 3]).strip())
        pool = blocks + [m for m in merged if m]
        best = max(pool, key=_candidate_score)
        method = "detector_merge" if best in merged else "detector_list"
        return best, method
    return "", "none"


def extract_kernel_with_method(text: str) -> tuple[Optional[str], str]:
    """Extract kernel code with method metadata for diagnostics."""
    if detector is None:
        return None, "detector_unavailable"

    tail = _extract_after_last_think(text).strip("\n")
    if not tail:
        return None, "empty_tail"

    blocks = [b.strip() for b in FENCED_BLOCK_RE.findall(tail) if b and b.strip()]
    if blocks:
        model_blocks = [b for b in blocks if ("class ModelNew" in b or "class Model " in b)]
        if model_blocks:
            return max(model_blocks, key=len), "fenced_pref"
        return max(blocks, key=len), "fenced_last"

    if _has_model_class(tail):
        prefix = _trim_to_compilable_prefix(tail)
        if prefix and _has_model_class(prefix):
            return prefix, "tail_prefix"

    best, method = _best_detector_candidate(detector(tail))
    if best:
        return best, method
    return None, "none"


def extract_kernel(text: str) -> Optional[str]:
    code, _ = extract_kernel_with_method(text)
    return code


def evaluate_kernel(kernel_code: Optional[str], reference: str) -> Dict[str, Any]:
    """Call kernelrl.eval.eval; return a dict with error info on failure."""
    if kernel_eval is None or kernel_code is None:
        return {"correct": False, "error": "detector_or_eval_not_available"}
    try:
        result = kernel_eval(kernel_code, reference)
        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}
        return result
    except Exception as exc:
        return {"correct": False, "error": str(exc), "traceback": traceback.format_exc()}


def evaluate_kernel_with_pipeline(
    kernel_code: Optional[str],
    reference: str,
    use_compile: bool = True,
    use_timing: bool = False,
    use_correctness: bool = True,
    use_hack_check: bool = True,
) -> Dict[str, Any]:
    """Call eval_pipeline for comprehensive kernel evaluation.

    Returns pipeline evaluation result with compilation, correctness, timing, and hack check.
    Falls back to simple eval if pipeline unavailable.
    """
    if run_eval_pipeline is None or kernel_code is None:
        return evaluate_kernel(kernel_code, reference)

    try:
        pipeline_request = {
            "reference_code": reference,
            "generated_code": kernel_code,
            "compile": {"code": kernel_code, "language": "python"} if use_compile else None,
            "correctness": {} if use_correctness else None,
            "timing": {} if use_timing else None,
            "hack_check": {"code": kernel_code} if use_hack_check else None,
            "gates": {
                "require_compile_ok": use_compile,
                "require_correctness_ok": use_correctness,
                "require_timing_ok": False,
                "require_hack_clean": use_hack_check,
            },
        }
        # Remove None values
        pipeline_request = {k: v for k, v in pipeline_request.items() if v is not None}

        result = run_eval_pipeline(pipeline_request)

        # Get correctness result
        correctness_metrics = result.get("summary", {}).get("correctness", {})
        correct = correctness_metrics.get("correct", False)

        return {
            "correct": correct,
            "pipeline_ok": result.get("ok", False),
            "pipeline_status": result.get("status"),
            "pipeline_result": result,
            # Include gate results for quick assessment
            "compile_ok": result.get("gate_results", {}).get("compile_gate", False),
            "correctness_ok": result.get("gate_results", {}).get("correctness_gate", False),
            "hack_clean": result.get("gate_results", {}).get("hack_gate", False),
        }
    except Exception as exc:
        return {
            "correct": False,
            "pipeline_ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _passes_success_gate(eval_result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(eval_result, dict):
        return False
    if any(k in eval_result for k in ("compile_ok", "correctness_ok", "hack_clean")):
        return bool(
            eval_result.get("compile_ok", False)
            and eval_result.get("correctness_ok", False)
            and eval_result.get("hack_clean", False)
        )
    return bool(eval_result.get("correct", False))


def _derive_sample_error(eval_result: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(eval_result, dict):
        return "missing_eval_result"
    if eval_result.get("error"):
        return str(eval_result.get("error"))
    if _passes_success_gate(eval_result):
        return None
    if eval_result.get("pipeline_result"):
        statuses = []
        if not eval_result.get("compile_ok", False):
            statuses.append("compile")
        if not eval_result.get("correctness_ok", False):
            statuses.append("correctness")
        if not eval_result.get("hack_clean", False):
            statuses.append("hack")
        if statuses:
            return "failed_" + "_".join(statuses)
        return str(eval_result.get("pipeline_status") or "pipeline_failed")
    if not eval_result.get("correct", False):
        return "evaluation_failed"
    return None


def _truncate_for_feedback(text: Any, max_chars: int) -> str:
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(truncated)"


def _build_feedback_message(
    prompt: str,
    kernel_code: str,
    eval_result: Optional[Dict[str, Any]],
    feedback_max_chars: int,
) -> str:
    snippet_max = max(200, int(feedback_max_chars)) if feedback_max_chars and feedback_max_chars > 0 else 3000
    pipeline_result = (eval_result or {}).get("pipeline_result", {}) if isinstance(eval_result, dict) else {}
    summary = pipeline_result.get("summary", {})
    modules = pipeline_result.get("modules", {})

    compile_summary = summary.get("compile", {})
    compile_module = modules.get("compile", {})
    compile_errors = compile_module.get("errors") or []
    compile_stderr = ((compile_module.get("artifacts") or {}).get("stderr") or "").strip()
    compile_first_error = ""
    if compile_errors:
        first_error = compile_errors[0]
        compile_first_error = (
            first_error.get("details", {}).get("first_error")
            or first_error.get("message")
            or first_error.get("id")
            or ""
        )
    if not compile_first_error and compile_stderr:
        compile_first_error = compile_stderr.splitlines()[-1]

    correctness_summary = summary.get("correctness", {})
    correctness_module = modules.get("correctness", {})
    correctness_error = ""
    correctness_errors = correctness_module.get("errors") or []
    correctness_issues = correctness_module.get("issues") or []
    if correctness_errors:
        correctness_error = (
            correctness_errors[0].get("message")
            or correctness_errors[0].get("id")
            or ""
        )
    elif correctness_issues:
        correctness_error = (
            correctness_issues[0].get("message")
            or correctness_issues[0].get("id")
            or ""
        )

    hack_module = modules.get("hack_check", {})
    hack_issues = (hack_module.get("issues") or [])[:3]
    hack_lines: List[str] = []
    for issue in hack_issues:
        issue_id = issue.get("id", "unknown")
        message = issue.get("message", "")
        hack_lines.append(f"- {issue_id}: {_truncate_for_feedback(message, min(snippet_max, 1000))}")
    if not hack_lines:
        hack_lines.append("- none")

    prompt_head = _truncate_for_feedback(prompt.strip().replace("\n", " "), 200)
    code_block = kernel_code if kernel_code else "# Empty extracted code"
    msg = (
        "You are fixing generated code for the same task.\n"
        f"Task restatement: {prompt_head}\n\n"
        "Current gate status:\n"
        f"- compile_ok: {bool((eval_result or {}).get('compile_ok', False))}\n"
        f"- correctness_ok: {bool((eval_result or {}).get('correctness_ok', False))}\n"
        f"- hack_clean: {bool((eval_result or {}).get('hack_clean', False))}\n\n"
        "Compile diagnostics:\n"
        f"- error_category: {compile_summary.get('error_category')}\n"
        f"- first_error: {_truncate_for_feedback(compile_first_error, snippet_max)}\n\n"
        "Correctness diagnostics:\n"
        f"- error_type: {correctness_summary.get('error_type')}\n"
        f"- message: {_truncate_for_feedback(correctness_error, snippet_max)}\n"
        f"- max_diff: {correctness_summary.get('max_diff')}\n"
        f"- mean_diff: {correctness_summary.get('mean_diff')}\n\n"
        "Hack issues (top 3):\n"
        + "\n".join(hack_lines)
        + "\n\n"
        "Previous extracted code:\n"
        "```python\n"
        f"{code_block}\n"
        "```\n\n"
        "Please return a full runnable Python solution only in code blocks, "
        "including ModelNew, get_inputs, and get_init_inputs."
    )
    return msg


def _messages_total_chars(messages: Sequence[Dict[str, str]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _compact_message_content(content: str, max_chars: int = 4000) -> str:
    compact = re.sub(
        r"```[\s\S]*?```",
        "```python\n# previous code omitted for context budget\n```",
        content,
    )
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "\n...(truncated for context budget)"


def prune_message_history(messages: Sequence[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """Keep initial prompt and newest turns while fitting context budget."""
    normalized = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
    ]
    if max_chars <= 0 or _messages_total_chars(normalized) <= max_chars:
        return normalized

    # Stage 1: drop oldest assistant+user turn pairs, keep initial prompt + newest context.
    while _messages_total_chars(normalized) > max_chars and len(normalized) > 3:
        del normalized[1:3]

    if _messages_total_chars(normalized) <= max_chars:
        return normalized

    # Stage 2: compact older message bodies (typically large code blocks in feedback/history).
    for i in range(1, max(1, len(normalized) - 2)):
        normalized[i]["content"] = _compact_message_content(normalized[i]["content"])
        if _messages_total_chars(normalized) <= max_chars:
            return normalized

    # Stage 3: hard trim non-initial messages oldest-first until budget is met.
    for i in range(1, len(normalized)):
        total = _messages_total_chars(normalized)
        if total <= max_chars:
            return normalized
        overflow = total - max_chars
        content = normalized[i]["content"]
        if not content:
            continue
        keep = max(200, len(content) - overflow)
        if keep < len(content):
            normalized[i]["content"] = content[:keep] + "\n...(truncated for context budget)"

    if _messages_total_chars(normalized) <= max_chars:
        return normalized

    # Final fallback: trim the newest message if still over budget.
    total = _messages_total_chars(normalized)
    if total > max_chars and len(normalized) >= 2:
        overflow = total - max_chars
        last = normalized[-1]["content"]
        keep = max(200, len(last) - overflow)
        if keep < len(last):
            normalized[-1]["content"] = last[:keep] + "\n...(truncated for context budget)"
    return normalized


def compute_item_summary(samples: List[SampleResult]) -> Dict[str, Any]:
    corrects = [bool(s.eval_result and s.eval_result.get("correct", False)) for s in samples]
    success_gates = [_passes_success_gate(s.eval_result) for s in samples]
    summary: Dict[str, Any] = {
        "num_samples": len(samples),
        "num_correct": sum(corrects),
        "num_success_gate": sum(success_gates),
        "pass@1": sum(corrects) / len(corrects) if corrects else 0.0,
        "success_gate@1": sum(success_gates) / len(success_gates) if success_gates else 0.0,
        "any_correct": any(corrects),
        "all_correct": all(corrects) if corrects else False,
        "any_success_gate": any(success_gates),
        "num_extracted": sum(1 for s in samples if s.kernel_code is not None),
        "errors": Counter(s.error for s in samples if s.error).most_common(),
    }
    speedups = [
        s.eval_result.get("speedup")
        for s in samples
        if s.eval_result and s.eval_result.get("speedup") is not None
    ]
    if speedups:
        summary["mean_speedup"] = sum(speedups) / len(speedups)
        summary["max_speedup"] = max(speedups)
    turn_counts = [len(s.turns or []) for s in samples if s.turns]
    if turn_counts:
        summary["mean_turns_used"] = sum(turn_counts) / len(turn_counts)
        summary["max_turns_used"] = max(turn_counts)
        stop_reason_counts: Counter[str] = Counter()
        success_turn_hist: Counter[int] = Counter()
        for sample in samples:
            turns = sample.turns or []
            if not turns:
                continue
            stop_reason = turns[-1].get("stop_reason")
            if stop_reason:
                stop_reason_counts[stop_reason] += 1
            if _passes_success_gate(sample.eval_result):
                success_turn_hist[len(turns)] += 1
        summary["stop_reason_counts"] = dict(stop_reason_counts)
        summary["success_turn_histogram"] = {str(k): v for k, v in sorted(success_turn_hist.items())}
    return summary


# ---------------------------------------------------------------------------
# Aggregated statistics
# ---------------------------------------------------------------------------
def compute_global_summary(
    results: Sequence[ItemResult],
    multi_turn_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total = len(results)
    any_correct = sum(1 for r in results if r.summary.get("any_correct", False))
    all_correct = sum(1 for r in results if r.summary.get("all_correct", False))
    total_samples = sum(r.summary.get("num_samples", 0) for r in results)
    total_correct = sum(r.summary.get("num_correct", 0) for r in results)
    total_extracted = sum(r.summary.get("num_extracted", 0) for r in results)

    global_summary: Dict[str, Any] = {
        "total_items": total,
        "total_samples": total_samples,
        "total_extracted": total_extracted,
        "any_correct_items": any_correct,
        "all_correct_items": all_correct,
        "overall_pass@1": total_correct / total_samples if total_samples else 0.0,
        "any_correct_rate": any_correct / total if total else 0.0,
    }
    all_speedups = [
        s.eval_result.get("speedup")
        for r in results
        for s in r.samples
        if s.eval_result and s.eval_result.get("speedup") is not None
    ]
    if all_speedups:
        global_summary["mean_speedup"] = sum(all_speedups) / len(all_speedups)
        global_summary["max_speedup"] = max(all_speedups)
    turn_counts: List[int] = []
    stop_reason_counts: Counter[str] = Counter()
    success_turn_hist: Counter[int] = Counter()
    for result in results:
        for sample in result.samples:
            turns = sample.turns or []
            if not turns:
                continue
            turn_counts.append(len(turns))
            stop_reason = turns[-1].get("stop_reason")
            if stop_reason:
                stop_reason_counts[stop_reason] += 1
            if _passes_success_gate(sample.eval_result):
                success_turn_hist[len(turns)] += 1
    if turn_counts:
        global_summary["mean_turns_used"] = sum(turn_counts) / len(turn_counts)
        global_summary["max_turns_used"] = max(turn_counts)
        global_summary["stop_reason_counts"] = dict(stop_reason_counts)
        global_summary["success_turn_histogram"] = {str(k): v for k, v in sorted(success_turn_hist.items())}
    if multi_turn_config:
        global_summary["multi_turn_config"] = dict(multi_turn_config)
    return global_summary


def print_summary(global_summary: Dict[str, Any], title: str = "Global Summary") -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for k, v in global_summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def save_summary_json(save_path: str, global_summary: Dict[str, Any]) -> None:
    summary_path = save_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------
def _is_evaluated(eval_result: Optional[Dict[str, Any]]) -> bool:
    """Check whether a single eval_result is valid (not a placeholder failure)."""
    if eval_result is None:
        return False
    err = eval_result.get("error")
    if err == "detector_or_eval_not_available":
        return False
    return True


def check_rollout_complete(records: Sequence[Dict[str, Any]], total: int) -> bool:
    """Check whether the rollout in save_path is complete (sufficient record count)."""
    return len(records) >= total


def check_eval_complete(records: Sequence[Dict[str, Any]]) -> bool:
    """Check whether evaluation is fully completed in the rollout results.

    Standard: every record has at least one sample, and all samples have a valid eval_result.
    """
    if not records:
        return False
    for rec in records:
        samples = rec.get("samples", [])
        if not samples:
            return False
        for s in samples:
            if not _is_evaluated(s.get("eval_result")):
                return False
    return True


# ---------------------------------------------------------------------------
# OpenAI-compatible API calls (async + concurrency control)
# ---------------------------------------------------------------------------
async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: Optional[str],
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_p: float,
    timeout: float,
    messages: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """Send a chat.completions request to the SGLang server and return n texts."""
    url = base_url.rstrip("/") + "/chat/completions"
    req_messages = messages if messages is not None else [{"role": "user", "content": prompt or ""}]
    payload = {
        "model": model,
        "messages": req_messages,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "top_p": top_p,
    }
    resp = await client.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    texts: List[str] = []
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content", "")
        texts.append(content)

    # Pad with empty strings if the server returns fewer than n choices
    while len(texts) < n:
        texts.append("")
    return texts


async def bounded_chat_completion(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    **kwargs: Any,
) -> List[str]:
    async with sem:
        return await chat_completion(client=client, **kwargs)


# ---------------------------------------------------------------------------
# Main processing flow
# ---------------------------------------------------------------------------
async def process_item(
    idx: int,
    prompt: str,
    reference: str,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    sample_id: Optional[int] = None,
    level: Optional[str] = None,
    difficulty: Optional[int] = None,
    file_path: Optional[str] = None,
) -> ItemResult:
    # 1) Request generation (turn 1)
    try:
        texts = await bounded_chat_completion(
            sem=sem,
            client=client,
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            top_p=args.top_p,
            timeout=args.timeout,
        )
    except Exception as exc:
        err_info = {
            "text": "",
            "kernel_code": None,
            "eval_result": None,
            "error": str(exc),
            "turns": [],
        }
        return ItemResult(
            idx=idx,
            sample_id=sample_id,
            prompt=prompt,
            reference=reference,
            level=level,
            difficulty=difficulty,
            file_path=file_path,
            samples=[SampleResult(**err_info) for _ in range(args.n)],
            summary={"all_failed": True, "error": str(exc)},
        )

    # Single-turn behavior unchanged unless --multi-turn is enabled.
    if not args.multi_turn:
        samples: List[SampleResult] = []
        for text in texts:
            kernel_code = extract_kernel(text)
            # Use pipeline evaluation if enabled, otherwise use standard evaluation
            if args.use_pipeline:
                eval_res = evaluate_kernel_with_pipeline(
                    kernel_code,
                    reference,
                    use_compile=args.pipeline_compile,
                    use_timing=args.pipeline_timing,
                    use_correctness=args.pipeline_correctness,
                    use_hack_check=args.pipeline_hack_check,
                )
            else:
                eval_res = evaluate_kernel(kernel_code, reference)
            samples.append(
                SampleResult(
                    text=text,
                    kernel_code=kernel_code,
                    eval_result=eval_res,
                    error=_derive_sample_error(eval_res),
                    turns=[],
                )
            )
        summary = compute_item_summary(samples)
        return ItemResult(
            idx=idx,
            sample_id=sample_id,
            prompt=prompt,
            reference=reference,
            level=level,
            difficulty=difficulty,
            file_path=file_path,
            samples=samples,
            summary=summary,
        )

    # Multi-turn mode: iterate each trajectory serially within this item.
    samples_mt: List[SampleResult] = []
    for text in texts:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        current_text = text
        turns: List[Dict[str, Any]] = []
        final_text = current_text
        final_code: Optional[str] = None
        final_eval: Optional[Dict[str, Any]] = None
        final_error: Optional[str] = None

        for turn_idx in range(1, max(1, args.max_turns) + 1):
            kernel_code, extract_method = extract_kernel_with_method(current_text)
            eval_res = evaluate_kernel_with_pipeline(
                kernel_code,
                reference,
                use_compile=True,
                use_timing=False,
                use_correctness=True,
                use_hack_check=True,
            )

            turn_rec: Dict[str, Any] = {
                "turn_idx": turn_idx,
                "assistant_text": current_text,
                "extract_method": extract_method,
                "kernel_code": kernel_code,
                "eval_result": eval_res,
                "feedback_to_model": None,
            }

            final_text = current_text
            final_code = kernel_code
            final_eval = eval_res
            final_error = _derive_sample_error(eval_res)

            if _passes_success_gate(eval_res):
                turn_rec["stop_reason"] = "success_gate"
                turns.append(turn_rec)
                break

            if turn_idx >= args.max_turns:
                turn_rec["stop_reason"] = "max_turns_reached"
                turns.append(turn_rec)
                break

            feedback = _build_feedback_message(
                prompt=prompt,
                kernel_code=kernel_code or "",
                eval_result=eval_res,
                feedback_max_chars=args.feedback_max_chars,
            )
            turn_rec["feedback_to_model"] = feedback
            turns.append(turn_rec)

            messages.append({"role": "assistant", "content": current_text})
            messages.append({"role": "user", "content": feedback})
            messages = prune_message_history(messages, max_chars=args.history_max_chars)

            try:
                next_texts = await bounded_chat_completion(
                    sem=sem,
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=None,
                    messages=messages,
                    n=1,
                    temperature=args.repair_temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed + turn_idx,
                    top_p=args.top_p,
                    timeout=args.timeout,
                )
                current_text = next_texts[0]
            except Exception as exc:
                final_error = str(exc)
                turn_rec["stop_reason"] = "generation_error"
                break

        samples_mt.append(
            SampleResult(
                text=final_text,
                kernel_code=final_code,
                eval_result=final_eval,
                error=final_error,
                turns=turns,
            )
        )

    summary = compute_item_summary(samples_mt)
    return ItemResult(
        idx=idx,
        sample_id=sample_id,
        prompt=prompt,
        reference=reference,
        level=level,
        difficulty=difficulty,
        file_path=file_path,
        samples=samples_mt,
        summary=summary,
    )


async def run_rollout(
    records: List[Dict[str, Any]],
    existing_results: List[ItemResult],
    args: argparse.Namespace,
) -> List[ItemResult]:
    """Execute rollout (with resume support)."""
    checkpoint_path = args.checkpoint_path or (args.save_path + ".checkpoint.jsonl")
    done_idxs = {r.idx for r in existing_results}
    pending_records = [(i, rec) for i, rec in enumerate(records) if i not in done_idxs]
    print(f"[INFO] Pending {len(pending_records)} / {len(records)} items")

    results = list(existing_results)
    if not pending_records:
        print("[INFO] All records already processed, skipping rollout")
        results.sort(key=lambda r: r.idx)
        return results

    sem = asyncio.Semaphore(args.max_concurrent)
    limits = httpx.Limits(
        max_keepalive_connections=args.max_concurrent,
        max_connections=args.max_concurrent * 2,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            process_item(
                idx=i,
                prompt=rec.get("prompt", rec.get("query", "")),
                reference=rec.get("reference_code", rec.get("reference", "")),
                sem=sem,
                client=client,
                args=args,
                sample_id=rec.get("sample_id"),
                level=rec.get("level"),
                difficulty=rec.get("difficulty"),
                file_path=rec.get("file_path"),
            )
            for i, rec in pending_records
        ]

        processed_since_checkpoint = 0
        if tqdm_async is not None:
            iterator = tqdm_async.as_completed(tasks, total=len(tasks), desc="Rollout")
        else:
            iterator = asyncio.as_completed(tasks)

        for coro in iterator:
            res = await coro
            results.append(res)
            done_idxs.add(res.idx)
            processed_since_checkpoint += 1

            if args.checkpoint_interval > 0 and processed_since_checkpoint >= args.checkpoint_interval:
                results.sort(key=lambda r: r.idx)
                save_jsonl(checkpoint_path, [item_result_to_dict(r) for r in results])
                print(f"[INFO] Checkpoint saved -> {checkpoint_path} ({len(results)} records)")
                processed_since_checkpoint = 0

    if args.checkpoint_interval > 0 and processed_since_checkpoint > 0:
        results.sort(key=lambda r: r.idx)
        save_jsonl(checkpoint_path, [item_result_to_dict(r) for r in results])
        print(f"[INFO] Final checkpoint saved -> {checkpoint_path} ({len(results)} records)")

    results.sort(key=lambda r: r.idx)
    return results


def run_eval_on_records(records: List[Dict[str, Any]], use_pipeline: bool = False, **pipeline_kwargs: Any) -> List[ItemResult]:
    """Re-run detector + eval on existing rollout records."""
    print(f"[INFO] Starting evaluation for {len(records)} records")
    results: List[ItemResult] = []
    record_iter = records
    if tqdm_sync is not None:
        record_iter = tqdm_sync(records, desc="Evaluate")

    for rec in record_iter:
        reference = rec.get("reference", "")
        samples: List[SampleResult] = []
        for s in rec.get("samples", []):
            text = s.get("text", "")
            kernel_code = extract_kernel(text)
            turns = s.get("turns") or []
            # Use pipeline evaluation if enabled
            if use_pipeline:
                eval_res = evaluate_kernel_with_pipeline(
                    kernel_code,
                    reference,
                    use_compile=pipeline_kwargs.get("pipeline_compile", True),
                    use_correctness=pipeline_kwargs.get("pipeline_correctness", True),
                    use_timing=pipeline_kwargs.get("pipeline_timing", False),
                    use_hack_check=pipeline_kwargs.get("pipeline_hack_check", True),
                )
            else:
                eval_res = evaluate_kernel(kernel_code, reference)
            samples.append(
                SampleResult(
                    text=text,
                    kernel_code=kernel_code,
                    eval_result=eval_res,
                    error=_derive_sample_error(eval_res),
                    turns=turns,
                )
            )
        summary = compute_item_summary(samples)
        results.append(
            ItemResult(
                idx=rec.get("idx", 0),
                sample_id=rec.get("sample_id"),
                prompt=rec.get("prompt", ""),
                reference=reference,
                level=rec.get("level"),
                difficulty=rec.get("difficulty"),
                file_path=rec.get("file_path"),
                samples=samples,
                summary=summary,
            )
        )
    return results


def save_final_results(results: List[ItemResult], args: argparse.Namespace, checkpoint_path: str) -> None:
    out_records = [item_result_to_dict(r) for r in results]
    save_jsonl(args.save_path, out_records)
    print(f"[INFO] Results saved to {args.save_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[INFO] Cleaned up checkpoint: {checkpoint_path}")

    multi_turn_config = None
    if args.multi_turn:
        multi_turn_config = {
            "enabled": True,
            "max_turns": args.max_turns,
            "repair_temperature": args.repair_temperature,
            "history_max_chars": args.history_max_chars,
            "feedback_max_chars": args.feedback_max_chars,
            "n": args.n,
        }
    gs = compute_global_summary(results, multi_turn_config=multi_turn_config)
    print_summary(gs)
    save_summary_json(args.save_path, gs)


async def main_async(args: argparse.Namespace) -> None:
    records = load_jsonl(args.data_path)
    total = len(records)
    print(f"[INFO] Loaded {total} records from {args.data_path}")

    save_path = args.save_path
    checkpoint_path = args.checkpoint_path or (save_path + ".checkpoint.jsonl")

    # Prepare pipeline kwargs if enabled
    pipeline_kwargs = {}
    if args.use_pipeline:
        pipeline_kwargs = {
            "pipeline_compile": args.pipeline_compile,
            "pipeline_correctness": args.pipeline_correctness,
            "pipeline_timing": args.pipeline_timing,
            "pipeline_hack_check": args.pipeline_hack_check,
        }
        print(f"[INFO] Using eval_pipeline with: compile={args.pipeline_compile}, correctness={args.pipeline_correctness}, timing={args.pipeline_timing}, hack_check={args.pipeline_hack_check}")

    # 1. Check whether save_path exists
    if os.path.exists(save_path):
        existing_records_raw = load_jsonl(save_path)
        existing_count = len(existing_records_raw)
        print(f"[INFO] Detected existing result file {save_path}, contains {existing_count}/{total} records")

        # 1a. Incomplete rollout -> resume rollout
        if existing_count < total:
            existing_results = [dict_to_item_result(r) for r in existing_records_raw]
            results = await run_rollout(records, existing_results, args)
            save_final_results(results, args, checkpoint_path)
            return

        # 1b. Rollout complete -> check whether evaluation is complete
        if check_eval_complete(existing_records_raw):
            print("[INFO] Evaluation already completed; printing summary directly")
            results = [dict_to_item_result(r) for r in existing_records_raw]
            multi_turn_config = None
            if args.multi_turn:
                multi_turn_config = {
                    "enabled": True,
                    "max_turns": args.max_turns,
                    "repair_temperature": args.repair_temperature,
                    "history_max_chars": args.history_max_chars,
                    "feedback_max_chars": args.feedback_max_chars,
                    "n": args.n,
                }
            gs = compute_global_summary(results, multi_turn_config=multi_turn_config)
            print_summary(gs)
            save_summary_json(save_path, gs)
            return

        # 1c. Rollout complete but eval incomplete -> re-evaluate
        print("[INFO] Rollout complete but evaluation incomplete; starting re-evaluation")
        results = run_eval_on_records(existing_records_raw, use_pipeline=args.use_pipeline, **pipeline_kwargs)
        save_final_results(results, args, checkpoint_path)
        return

    # 2. save_path does not exist -> start from scratch
    print(f"[INFO] Result file {save_path} does not exist, starting rollout from scratch")
    results = await run_rollout(records, [], args)
    save_final_results(results, args, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-turn/multi-turn rollout + evaluation for kernelbench tasks. "
                    "Automatically resumes from save-path if it exists."
    )
    parser.add_argument("--data-path", required=True, help="Path to input JSONL file")
    parser.add_argument("--save-path", required=True, help="Core save path; results/checkpoint/summary are derived from this")
    parser.add_argument("--base-url", default="http://localhost:30000/v1", help="SGLang server OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="Model name (must match the model loaded by the SGLang server)")
    parser.add_argument("--n", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=81920, help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Maximum concurrent requests")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save a checkpoint every N processed items (<=0 disables intermediate checkpoints)",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint file path; defaults to {save-path}.checkpoint.jsonl")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        default=False,
        help="Enable multi-turn repair loop: generate -> evaluate -> feedback -> regenerate",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum turns per trajectory in multi-turn mode",
    )
    parser.add_argument(
        "--repair-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for repair turns (turn>=2)",
    )
    parser.add_argument(
        "--history-max-chars",
        type=int,
        default=180000,
        help="Max chars for chat history in multi-turn mode before pruning oldest turns",
    )
    parser.add_argument(
        "--feedback-max-chars",
        type=int,
        default=3000,
        help="Max chars for each structured feedback message in multi-turn mode",
    )

    # Evaluation pipeline options
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        default=False,
        help="Use kernelrl.eval.pipeline for comprehensive evaluation (compile, correctness, timing, hack check)",
    )
    parser.add_argument(
        "--pipeline-compile",
        action="store_true",
        default=True,
        help="Enable compilation checking in pipeline (default: True)",
    )
    parser.add_argument(
        "--no-pipeline-compile",
        action="store_false",
        dest="pipeline_compile",
        help="Disable compilation checking in pipeline",
    )
    parser.add_argument(
        "--pipeline-correctness",
        action="store_true",
        default=True,
        help="Enable correctness checking in pipeline (default: True)",
    )
    parser.add_argument(
        "--no-pipeline-correctness",
        action="store_false",
        dest="pipeline_correctness",
        help="Disable correctness checking in pipeline",
    )
    parser.add_argument(
        "--pipeline-timing",
        action="store_true",
        default=False,
        help="Enable timing/benchmarking in pipeline (default: False)",
    )
    parser.add_argument(
        "--pipeline-hack-check",
        action="store_true",
        default=True,
        help="Enable hack/security checking in pipeline (default: True)",
    )
    parser.add_argument(
        "--no-pipeline-hack-check",
        action="store_false",
        dest="pipeline_hack_check",
        help="Disable hack/security checking in pipeline",
    )

    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be greater than 0")
    if args.max_turns <= 0:
        parser.error("--max-turns must be greater than 0")
    if args.history_max_chars <= 0:
        parser.error("--history-max-chars must be greater than 0")
    if args.multi_turn:
        # Multi-turn always uses compile+correctness+hack feedback loop.
        args.use_pipeline = True
        args.pipeline_compile = True
        args.pipeline_correctness = True
        args.pipeline_hack_check = True
        args.pipeline_timing = False

    start = time.time()
    asyncio.run(main_async(args))
    elapsed = time.time() - start
    print(f"[INFO] All done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
