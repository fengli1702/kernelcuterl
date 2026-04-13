#!/usr/bin/env python3
"""
Re-evaluate V9 outputs by re-extracting code from the text after </think>.
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

from kernelrl.eval.pipeline import run_eval_pipeline
from kernelrl.eval.detector import detector


CLOSE_TAG_RE = re.compile(r"(?is)</think>|</thinking>|</Think>|</Thought>|</THINK>|</THINKING>|</thinking>")
MODEL_CLASS_RE = re.compile(r"(?m)^\s*class\s+Model(?:New)?\b")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_after_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = CLOSE_TAG_RE.search(text)
    if not m:
        return text
    return text[m.end():]


def _has_model_class(code: str) -> bool:
    return bool(code and MODEL_CLASS_RE.search(code))


def _compile_status(code: str):
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


def _candidate_score(code: str) -> tuple:
    ok, _ = _compile_status(code)
    return (
        1 if _has_model_class(code) else 0,
        1 if "def forward" in code else 0,
        1 if "def get_inputs" in code else 0,
        1 if "def get_init_inputs" in code else 0,
        1 if ok else 0,
        len(code),
    )


def _best_detector_candidate(cand) -> tuple[str, str]:
    if cand is None:
        return "", "none"
    if isinstance(cand, str):
        text = cand.strip()
        return (text, "detector_single") if text else ("", "none")
    if isinstance(cand, (list, tuple)):
        blocks = [str(c).strip() for c in cand if str(c).strip()]
        if not blocks:
            return "", "none"
        merged = []
        for i in range(len(blocks) - 1):
            merged.append("\n\n".join(blocks[i : i + 2]).strip())
        for i in range(len(blocks) - 2):
            merged.append("\n\n".join(blocks[i : i + 3]).strip())
        pool = blocks + [m for m in merged if m]
        best = max(pool, key=_candidate_score)
        method = "detector_merge" if best in merged else "detector_list"
        return best, method
    return "", "none"


def extract_code_from_text(text: str):
    tail = extract_after_think(text)
    if not tail:
        return "", "empty_tail"
    tail = tail.strip("\n")
    if not tail:
        return "", "empty_tail"

    # 1) fenced code blocks in the tail
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+#-]*)?\n?(.*?)\n?```", tail, re.DOTALL)
    blocks = [b.strip() for b in blocks if b and b.strip()]
    if blocks:
        # Prefer blocks that contain model class definitions.
        model_blocks = [b for b in blocks if ("class ModelNew" in b or "class Model " in b)]
        if model_blocks:
            return max(model_blocks, key=len), "fenced_pref"

        # fallback: choose the largest fenced block
        return max(blocks, key=len), "fenced_last"

    # 2) no fenced block: try raw tail (keep indentation) and trim trailing noise.
    if _has_model_class(tail):
        prefix = _trim_to_compilable_prefix(tail)
        if prefix and _has_model_class(prefix):
            return prefix, "tail_prefix"

    # 3) fallback to detector heuristics (edge cases)
    cand = detector(tail)
    best, method = _best_detector_candidate(cand)
    return best, method


def evaluate_item(payload):
    idx, record, gpu_id, correctness_timeout = payload
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sample = record["samples"][0]
    text = sample.get("text", "")
    code, method = extract_code_from_text(text)
    code = code or ""

    request = {
        "reference_code": record.get("reference", ""),
        "generated_code": code,
        "compile": {"code": code, "language": "python"},
        "correctness": {"timeout_sec": correctness_timeout},
        "gates": {
            "require_compile_ok": True,
            "require_correctness_ok": True,
            "require_timing_ok": False,
            "require_hack_clean": False,
        },
        "hack_check": {"code": code},
    }

    if code == "":
        # keep request minimal to avoid odd parser behavior
        request.pop("compile")
        request.pop("gates")
        request.pop("hack_check", None)
        request["compile"] = {}

    result = run_eval_pipeline(request)

    module_compile = result.get("modules", {}).get("compile", {})
    module_correct = result.get("modules", {}).get("correctness", {})

    compile_errors = module_compile.get("errors") or []
    compile_issues = module_compile.get("issues") or []
    correctness_errors = module_correct.get("errors") or []
    correctness_issues = module_correct.get("issues") or []

    compile_ok = bool(result.get("gate_results", {}).get("compile_gate", False))
    correctness_ok = bool(result.get("gate_results", {}).get("correctness_gate", False))

    def first_id(items):
        return items[0].get("id") if items else None

    out = {
        "sample_id": record.get("sample_id"),
        "idx": record.get("idx", idx),
        "extract_method": method,
        "extract_len": len(code),
        "compile_ok": compile_ok,
        "compile_status": module_compile.get("status"),
        "compile_error": first_id(compile_errors),
        "compile_issue": first_id(compile_issues),
        "compile_return_code": module_compile.get("metrics", {}).get("return_code"),
        "correctness_ok": correctness_ok,
        "correctness_status": module_correct.get("status"),
        "correctness_error": first_id(correctness_errors),
        "correctness_issue": first_id(correctness_issues),
        "pipeline_ok": bool(result.get("ok", False)),
        "pipeline_status": result.get("status"),
        "text": text,
        "kernel_code": code,
        "eval_result": {
            "correct": result.get("ok") if "ok" in result else False,
            "pipeline_ok": result.get("ok", False),
            "pipeline_status": result.get("status", ""),
            "pipeline_result": result,
        },
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rollout/results/kernelbench_rollout_v9_python.jsonl")
    parser.add_argument("--output", default="rollout/results/kernelbench_rollout_v9_think_correctness_multigpu.jsonl")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--correctness-timeout", type=int, default=120)
    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"Loaded {len(items)} items")

    n = len(items)
    n_gpu = args.num_gpus
    gpu_items = []
    per = n // n_gpu
    for i in range(n_gpu):
        start = i * per
        end = (i + 1) * per if i < n_gpu - 1 else n
        for item in items[start:end]:
            gpu_items.append((item.get("idx", 0), item, i, args.correctness_timeout))

    with multiprocessing.Pool(processes=n_gpu) as pool:
        outputs = pool.map(evaluate_item, gpu_items)

    outputs.sort(key=lambda x: x["idx"] if x["idx"] is not None else 0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summary
    total = len(outputs)
    compile_ok = sum(1 for x in outputs if x["compile_ok"])
    compile_fail = total - compile_ok
    correctness_ok = sum(1 for x in outputs if x["correctness_ok"] and x["compile_ok"])
    correctness_fail = sum(1 for x in outputs if x["compile_ok"] and not x["correctness_ok"])
    correctness_skip = sum(1 for x in outputs if not x["compile_ok"])

    summary = {
        "total": total,
        "correctness_timeout_sec": args.correctness_timeout,
        "has_think_marker": sum(
            1
            for item in items
            if bool(CLOSE_TAG_RE.search(item.get("samples", [{}])[0].get("text", "")))
        ),
        "compile_ok": compile_ok,
        "compile_fail": compile_fail,
        "correctness_ok": correctness_ok,
        "correctness_fail": correctness_fail,
        "correctness_skip": correctness_skip,
    }
    with out_path.with_suffix('.summary.json').open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(summary)


if __name__ == "__main__":
    main()
