# Ref Smoke Summary (simplified_v2)

- Dataset: `data/release/final_v1_20260417/kernel_dataset_final_v1_simplified_v2.jsonl`
- Runner: `scripts/ref_smoke_check.py`
- Runtime env: `nvcr.io/nvidia/pytorch:25.12-py3` with `--gpus all`
- Sampling: `per-source=3`, `timeout=25s`
- Full report: `data/release/final_v1_20260417/ref_smoke_check_simplified_v2_per_source3.json`

## Summary

- Total checked: `21`
- OK: `18`
- Fail: `3`

## By Source

- `kernelbench`: 3/3
- `kernelbook`: 2/3
- `cuda_agent_ops_6k`: 3/3
- `drkernel`: 3/3
- `tritonbench`: 1/3
- `multikernelbench`: 3/3
- `xpuoj`: 3/3

## Format Check (sample per source)

All sampled sources contain standard interface symbols:
- `Model`
- `ModelNew`
- `get_inputs`
- `get_init_inputs`
- `run_kernel`

## Failed Samples

1. `kernelbook__976bb64b0712842c`
- Error: `inference_failed: _LinAlgError: cholesky ... input is not positive-definite`
- Meaning: numerical property of sampled input (not adapter format issue)

2. `tritonbench__41474e4de6cf1a77`
- Error: `Legacy autograd function with non-static forward method is deprecated`
- Meaning: upstream code compatibility issue with current torch version

3. `tritonbench__8f57d36080bd3b44`
- Error: `Cannot call @triton.jit'd outside of the scope of a kernel`
- Meaning: upstream Triton invocation style issue

