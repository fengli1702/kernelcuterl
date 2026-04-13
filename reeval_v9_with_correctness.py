#!/usr/bin/env python3
"""
Re-evaluate V9 results with correctness checking enabled.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rollout.single_turn_rollout import (
    evaluate_kernel_with_pipeline,
    extract_kernel,
    compute_global_summary,
    save_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reeval_v9_results(input_path: str, output_path: str):
    """Re-evaluate V9 results with correctness checking."""
    print(f"[INFO] Loading V9 results from {input_path}")
    v9_results = load_jsonl(input_path)
    print(f"[INFO] Loaded {len(v9_results)} items")

    print("[INFO] Re-evaluating with correctness checking enabled...")

    new_results = []
    items_iter = v9_results
    if tqdm is not None:
        items_iter = tqdm(v9_results, desc="Re-evaluating")

    correct_count = 0

    for item in items_iter:
        reference = item.get("reference", "")
        samples = item.get("samples", [])

        new_samples = []
        for sample in samples:
            text = sample.get("text", "")
            kernel_code = sample.get("kernel_code")

            # If kernel_code not extracted yet, extract it
            if kernel_code is None:
                kernel_code = extract_kernel(text)

            # Re-evaluate with correctness checking
            eval_result = evaluate_kernel_with_pipeline(
                kernel_code=kernel_code,
                reference=reference,
                use_compile=True,
                use_correctness=True,
                use_timing=False,
                use_hack_check=False,
            )

            # Track correctness
            if eval_result.get("correct"):
                correct_count += 1

            new_sample = {
                "text": text,
                "kernel_code": kernel_code,
                "eval_result": eval_result,
                "error": eval_result.get("error") if not eval_result.get("correct") else None,
            }
            new_samples.append(new_sample)

        # Compute item summary
        corrects = [bool(s["eval_result"] and s["eval_result"].get("correct", False)) for s in new_samples]
        summary = {
            "num_samples": len(new_samples),
            "num_correct": sum(corrects),
            "pass@1": sum(corrects) / len(corrects) if corrects else 0.0,
            "any_correct": any(corrects),
            "all_correct": all(corrects) if corrects else False,
            "num_extracted": sum(1 for s in new_samples if s["kernel_code"] is not None),
        }

        new_item = {
            "idx": item.get("idx"),
            "sample_id": item.get("sample_id"),
            "file_path": item.get("file_path"),
            "level": item.get("level"),
            "difficulty": item.get("difficulty"),
            "prompt": item.get("prompt"),
            "reference": reference,
            "samples": new_samples,
            "summary": summary,
        }
        new_results.append(new_item)

    # Save results
    print(f"[INFO] Saving results to {output_path}")
    save_jsonl(output_path, new_results)

    # Compute and print global summary
    print("\n" + "=" * 60)
    print("Global Summary")
    print("=" * 60)

    total_samples = len(v9_results)
    any_correct = sum(1 for r in new_results if r["summary"].get("any_correct", False))
    all_correct = sum(1 for r in new_results if r["summary"].get("all_correct", False))
    total_extracted = sum(r["summary"].get("num_extracted", 0) for r in new_results)

    print(f"  total_items: {len(new_results)}")
    print(f"  total_samples: {total_samples}")
    print(f"  total_extracted: {total_extracted}")
    print(f"  total_correct: {correct_count}")
    print(f"  any_correct_items: {any_correct}")
    print(f"  all_correct_items: {all_correct}")
    print(f"  overall_pass@1: {correct_count / total_samples if total_samples else 0.0:.2%}")
    print(f"  any_correct_rate: {any_correct / len(new_results) if new_results else 0.0:.2%}")
    print("=" * 60)

    # Save summary
    summary_path = output_path + ".summary.json"
    global_summary = {
        "total_items": len(new_results),
        "total_samples": total_samples,
        "total_extracted": total_extracted,
        "total_correct": correct_count,
        "any_correct_items": any_correct,
        "all_correct_items": all_correct,
        "overall_pass@1": correct_count / total_samples if total_samples else 0.0,
        "any_correct_rate": any_correct / len(new_results) if new_results else 0.0,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    input_path = "rollout/results/kernelbench_rollout_v9_python.jsonl"
    output_path = "rollout/results/kernelbench_rollout_v9_correctness.jsonl"

    reeval_v9_results(input_path, output_path)
    print("\n[INFO] Done!")
