#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_pipeline.pipeline import run_eval_pipeline


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo: torch.cuda.is_available() is false")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    m = 1024
    k = 1024
    n = 1024
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    out_ref = torch.empty(m, n, device=device, dtype=torch.float32)
    out_gen = torch.empty(m, n, device=device, dtype=torch.float32)

    def reference_fn() -> None:
        torch.mm(a, b, out=out_ref)

    def generated_fn() -> None:
        torch.mm(a, b, out=out_gen)

    reference_code = (
        "import torch\n\n"
        "def forward(a, b):\n"
        "    return torch.mm(a, b)\n"
    )
    generated_code = (
        "import torch\n\n"
        "def forward(a, b):\n"
        "    return torch.mm(a, b)\n"
        "...<truncated>..."
    )

    req = {
        "sample_id": "demo_cudagraph_truncation_fallback",
        "reference_code": reference_code,
        "generated_code": generated_code,
        "use_reference_on_generated_truncation": True,
        "generated_code_min_len": 48,
        "compile": {
            "language": "python",
            "timeout_sec": 30,
        },
        "hack_check": {
            "strict": True,
            "torch_usage_policy": "warn",
            "precision_target": "fp32",
        },
        "timing": {
            "mode": "cudagraph",
            "device": str(device),
            "generated_fn": generated_fn,
            "reference_fn": reference_fn,
            "warmup": 10,
            "warmup_ms": 100.0,
            "graph_iters": 20,
            "pre_capture_iters": 3,
            "trial_count": 7,
            "min_measure_ms": 200.0,
            "min_replays": 5,
            "max_replays": 100000,
            "probe_calls": 20,
            "suspicious_ratio_threshold": 0.25,
            "allow_suspicious": True,
            "require_not_suspicious": False,
            "use_reference_fn_on_generated_truncation": True,
            "allow_without_compile": True,
        },
        "gates": {
            "require_compile_ok": True,
            "require_timing_ok": True,
            "require_hack_clean": False,
        },
    }

    out = run_eval_pipeline(req)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
