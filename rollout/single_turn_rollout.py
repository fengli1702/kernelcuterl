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


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    text: str
    kernel_code: Optional[str]
    eval_result: Optional[Dict[str, Any]]
    error: Optional[str] = None


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


def extract_kernel(text: str) -> Optional[str]:
    """Call kernelrl.eval.detector and perform basic post-processing."""
    if detector is None:
        return None
    try:
        code = detector(text)
        # If a list/tuple is returned, pick the first non-empty element
        if isinstance(code, (list, tuple)):
            for c in code:
                if c is not None and str(c).strip():
                    return str(c).strip()
            return None
        if code is None or (isinstance(code, str) and not code.strip()):
            return None
        return str(code).strip()
    except Exception as exc:
        return None


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


def compute_item_summary(samples: List[SampleResult]) -> Dict[str, Any]:
    corrects = [bool(s.eval_result and s.eval_result.get("correct", False)) for s in samples]
    summary: Dict[str, Any] = {
        "num_samples": len(samples),
        "num_correct": sum(corrects),
        "pass@1": sum(corrects) / len(corrects) if corrects else 0.0,
        "any_correct": any(corrects),
        "all_correct": all(corrects) if corrects else False,
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
    return summary


# ---------------------------------------------------------------------------
# Aggregated statistics
# ---------------------------------------------------------------------------
def compute_global_summary(results: Sequence[ItemResult]) -> Dict[str, Any]:
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
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_p: float,
    timeout: float,
) -> List[str]:
    """Send a chat.completions request to the SGLang server and return n texts."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    # 1) Request generation
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
        err_info = {"text": "", "kernel_code": None, "eval_result": None, "error": str(exc)}
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

    # 2) Extract + evaluate
    samples: List[SampleResult] = []
    for text in texts:
        kernel_code = extract_kernel(text)
        eval_res = evaluate_kernel(kernel_code, reference)
        samples.append(
            SampleResult(
                text=text,
                kernel_code=kernel_code,
                eval_result=eval_res,
                error=eval_res.get("error") if eval_res and not eval_res.get("correct", False) else None,
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


def run_eval_on_records(records: List[Dict[str, Any]]) -> List[ItemResult]:
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
            eval_res = evaluate_kernel(kernel_code, reference)
            samples.append(
                SampleResult(
                    text=text,
                    kernel_code=kernel_code,
                    eval_result=eval_res,
                    error=eval_res.get("error") if eval_res and not eval_res.get("correct", False) else None,
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

    gs = compute_global_summary(results)
    print_summary(gs)
    save_summary_json(args.save_path, gs)


async def main_async(args: argparse.Namespace) -> None:
    records = load_jsonl(args.data_path)
    total = len(records)
    print(f"[INFO] Loaded {total} records from {args.data_path}")

    save_path = args.save_path
    checkpoint_path = args.checkpoint_path or (save_path + ".checkpoint.jsonl")

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
            gs = compute_global_summary(results)
            print_summary(gs)
            save_summary_json(save_path, gs)
            return

        # 1c. Rollout complete but eval incomplete -> re-evaluate
        print("[INFO] Rollout complete but evaluation incomplete; starting re-evaluation")
        results = run_eval_on_records(existing_records_raw)
        save_final_results(results, args, checkpoint_path)
        return

    # 2. save_path does not exist -> start from scratch
    print(f"[INFO] Result file {save_path} does not exist, starting rollout from scratch")
    results = await run_rollout(records, [], args)
    save_final_results(results, args, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-turn rollout + evaluation for kernelbench tasks. "
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
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be greater than 0")

    start = time.time()
    asyncio.run(main_async(args))
    elapsed = time.time() - start
    print(f"[INFO] All done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
