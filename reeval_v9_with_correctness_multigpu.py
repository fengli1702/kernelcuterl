#!/usr/bin/env python3
"""
Re-evaluate V9 results with correctness checking enabled - Multi-GPU version.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from multiprocessing import Pool, Manager
from functools import partial

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rollout.single_turn_rollout import (
    evaluate_kernel_with_pipeline,
    extract_kernel,
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


def evaluate_single_item(item: Dict[str, Any], gpu_id: int, progress_dict: Dict) -> Dict[str, Any]:
    """Evaluate a single item on a specific GPU."""
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

    # Update progress
    progress_dict['completed'] += 1

    return new_item


def evaluate_worker(items_and_gpu: tuple, progress_dict: Dict) -> List[Dict[str, Any]]:
    """Worker function for multiprocessing."""
    items, gpu_id = items_and_gpu
    results = []

    for item in items:
        result = evaluate_single_item(item, gpu_id, progress_dict)
        results.append(result)

    return results


def reeval_v9_results_multi_gpu(input_path: str, output_path: str, num_gpus: int = 8):
    """Re-evaluate V9 results with correctness checking using multiple GPUs."""
    print(f"[INFO] Loading V9 results from {input_path}")
    v9_results = load_jsonl(input_path)
    print(f"[INFO] Loaded {len(v9_results)} items")
    print(f"[INFO] Using {num_gpus} GPUs for parallel evaluation")

    # Split items among GPUs
    items_per_gpu = len(v9_results) // num_gpus
    gpu_items = []
    for i in range(num_gpus):
        start_idx = i * items_per_gpu
        if i == num_gpus - 1:
            # Last GPU gets remaining items
            end_idx = len(v9_results)
        else:
            end_idx = (i + 1) * items_per_gpu

        gpu_items.append((v9_results[start_idx:end_idx], i))

    print(f"[INFO] Items per GPU: {[len(items) for items, _ in gpu_items]}")
    print("[INFO] Starting multi-GPU re-evaluation...")

    # Create shared progress counter
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0

    # Use multiprocessing to parallelize
    worker_fn = partial(evaluate_worker, progress_dict=progress_dict)

    with Pool(processes=num_gpus) as pool:
        # Monitor progress
        if tqdm is not None:
            print("[INFO] Starting evaluation with progress bar...")

        # Execute in parallel
        results_per_gpu = pool.map(worker_fn, gpu_items)

    # Flatten results
    print("\n[INFO] Collecting results...")
    new_results = []
    for gpu_results in results_per_gpu:
        new_results.extend(gpu_results)

    # Sort by idx to maintain order
    new_results.sort(key=lambda x: x.get("idx", 0))

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

    # Count total correct samples
    correct_count = 0
    for r in new_results:
        for s in r["samples"]:
            if s.get("eval_result", {}).get("correct"):
                correct_count += 1

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
    import argparse

    parser = argparse.ArgumentParser(description="Re-evaluate V9 with correctness checking using multiple GPUs")
    parser.add_argument("--input", default="rollout/results/kernelbench_rollout_v9_python.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="rollout/results/kernelbench_rollout_v9_correctness.jsonl", help="Output JSONL file")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")

    args = parser.parse_args()

    reeval_v9_results_multi_gpu(args.input, args.output, args.num_gpus)
    print("\n[INFO] Done!")
