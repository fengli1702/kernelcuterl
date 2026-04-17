# KernelBench Format Status And Checks (2026-04-17)

## 1. Goal (Target Data Contract)

Our target is to make each sample evaluable with a KernelBench-style interface:

1. `reference_artifacts.reference_code` must be executable Python.
2. It must provide:
   - `class Model(nn.Module)`
   - `def get_inputs() -> list`
   - `def get_init_inputs() -> list`
3. Evaluator compares:
   - reference `Model`
   - generated `ModelNew` (or adapted class)
   - using `get_inputs/get_init_inputs` from reference side.
4. Query text should be consistent with this evaluator contract (no contradictory entry-point requirements).

## 2. Current Dataset Status (v6 Strict)

Dataset file:

- `/home/i_lidaifeng/dataset/data/processed/canonical_jsonl/canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar_semantic_ir_embedding_dedup_decontam_eval_model_iface_v6_strict.jsonl`

Scale:

- total: `20586`
- source split: `train=20297`, `eval=289`
- source distribution:
  - `kernelbook=9092`
  - `drkernel=6677`
  - `cuda_agent_ops_6k=4350`
  - `kernelbench=244`
  - `tritonbench=178`
  - `xpuoj=23`
  - `multikernelbench=22`
- backend:
  - `triton=15960`
  - `cuda=4626`

Removed sources in this release:

- `computeeval`
- `cudabench`

Evidence file:

- `dataset_shards/release_20260416_v6/filter_drop_cudabench_computeeval_20260416.json`

## 3. Interface Readiness Snapshot

Static scan results on the v6 strict JSONL:

1. All samples have non-empty reference code:
   - `reference_artifacts.reference_code`: `20586/20586`
2. In reference code, all samples include required KernelBench symbols:
   - `class Model(...)`: `20586/20586`
   - `def get_inputs(...)`: `20586/20586`
   - `def get_init_inputs(...)`: `20586/20586`
3. Static mismatch check (`forward` required args vs `get_inputs=[]`, `__init__` required args vs `get_init_inputs=[]`) found no direct contradiction by AST heuristic.

Empty-list getters still exist in some samples, but currently no AST-level required-arg mismatch was detected.

## 4. Gaps Relative To The Goal

### 4.1 Query Contract Is Not Fully Unified

Current query templates still include variants that ask for:

- `ModelNew`
- `run_kernel(*args, **kwargs)`

This can conflict with a pure KernelBench evaluator that expects only `Model`-style contract.

### 4.2 Query Content Coverage Is Uneven

From scan of `query` text:

- `query` containing `class Model(`: `11333/20586`
- `query` containing `get_inputs`: `20385/20586`
- `query` containing `get_init_inputs`: `20385/20586`

Main missing portion is concentrated in `tritonbench` and `xpuoj` query text variants (reference code still has wrappers).

### 4.3 Metadata Can Be Misleading

`reference_modality` still reports:

- `pytorch=20385`
- `text=201`

But those `text` rows already have executable code in `reference_artifacts.reference_code`.

## 5. Known Runtime Risk: Host GPU Stack

Current environment has host-side GPU runtime instability symptoms:

1. CPU-only Docker run works.
2. Docker with `--gpus` hangs/fails (`context canceled`).
3. Host shows many stuck processes in uninterruptible state (`D/Dl`):
   - `nvidia-smi -q -d compute`
   - `nvidia-container-runtime-hook prestart`
   - `nvidia-container-cli --load-kmods ... configure`

Impact:

- full GPU ref smoke checks are not trustworthy until host GPU runtime is recovered.

## 6. How To Check Problems (Operational Checklist)

### 6.1 Data Contract Check (No GPU Needed)

Run on v6 strict JSONL:

```bash
python3 - <<'PY'
import json, re
from collections import Counter
path='/home/i_lidaifeng/dataset/data/processed/canonical_jsonl/canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar_semantic_ir_embedding_dedup_decontam_eval_model_iface_v6_strict.jsonl'
c=Counter()
for line in open(path,encoding='utf-8'):
    row=json.loads(line)
    ref=(row.get('reference_artifacts') or {}).get('reference_code') or ''
    q=row.get('query') or ''
    c['total']+=1
    c['has_ref']+=bool(ref.strip())
    c['ref_Model']+=bool(re.search(r'(^|\\n)\\s*class\\s+Model\\s*\\(', ref))
    c['ref_get_inputs']+=bool(re.search(r'(^|\\n)\\s*def\\s+get_inputs\\s*\\(', ref))
    c['ref_get_init_inputs']+=bool(re.search(r'(^|\\n)\\s*def\\s+get_init_inputs\\s*\\(', ref))
    c['q_Model']+=bool(re.search(r'(^|\\n)\\s*class\\s+Model\\s*\\(', q))
print(dict(c))
PY
```

Pass criteria:

- `has_ref == total`
- `ref_Model == total`
- `ref_get_inputs == total`
- `ref_get_init_inputs == total`

### 6.2 GPU Runtime Health Check (Before Any Full Smoke Run)

```bash
timeout 10s docker run --rm ubuntu:22.04 bash -lc 'echo no_gpu_ok'
timeout 15s docker run --rm --gpus device=0 ubuntu:22.04 bash -lc 'echo gpu_ok'
```

Interpretation:

- first command ok + second command hang/fail => host GPU runtime problem, not dataset problem.

### 6.3 Full Reference Smoke Check (Only After 6.2 Passes)

1. Use shard-based run.
2. Record per-source failures.
3. Group by error signature:
   - timeout
   - CUDA busy/unavailable
   - Triton context/API mismatch
   - missing helper dependency
   - numeric/operator behavior mismatch

### 6.4 Required Failure Report Format

For each failed sample include:

- `sample_id`
- `source_name`
- `error_cluster`
- `first_error_line`
- `traceback_head`
- `can_auto_fix` (`yes/no`)
- `suggested_fix`

## 7. Typical Problem Types And What They Mean

1. `worker_timeout`:
   - may be heavy kernel or runtime hang.
   - action: increase timeout first; if still failing, inspect kernel launch or infinite loop.
2. `CUDA device busy/unavailable`:
   - environment contention.
   - action: isolate GPU, serialize jobs, retest.
3. `Triton context` errors:
   - version/API mismatch or launch signature mismatch.
   - action: pin Triton/PyTorch version and patch call sites.
4. helper import issues (for legacy/paritybench samples):
   - environment dependency issue.
   - action: vendor helper or rewrite reference sample to self-contained form.

## 8. Release Gate (Must Pass Before Final Packaging)

1. Data contract gate:
   - all rows pass `Model/get_inputs/get_init_inputs` check on reference code.
2. Query consistency gate:
   - query instructions aligned with evaluator entry contract.
3. Runtime gate:
   - host GPU health check passes.
4. Smoke gate:
   - full run completed with failure list generated and triaged.
5. Repro gate:
   - commit hash, report path, and command lines are recorded.

## 9. Current Recommended Next Step

1. First fix host GPU runtime (admin action required: restart docker + nvidia stack).
2. Re-run full ref smoke check on all `20586` samples in Docker GPU mode.
3. Patch failures by error cluster.
4. Unify query wording strictly to the selected evaluator interface (KernelBench-only contract).

## 10. Environment Requirements

### 10.1 Dataset processing only

- Linux (`x86_64`)
- Python `>=3.10`
- `bash` + coreutils (`cat`, `split`, `sha256sum`, `wc`)

### 10.2 Full runtime smoke/eval (GPU)

- Docker `>=24`
- NVIDIA Container Toolkit
- NVIDIA driver/NVML healthy (`nvidia-smi` must return quickly)
- CUDA GPU resources available (not busy/unavailable)
- Python `>=3.10`
- PyTorch/Triton versions aligned with the evaluator code

Quick preflight:

```bash
timeout 10s docker run --rm ubuntu:22.04 bash -lc 'echo no_gpu_ok'
timeout 15s docker run --rm --gpus device=0 ubuntu:22.04 bash -lc 'echo gpu_ok'
```
