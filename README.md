# KernelRL

```bash
MODEL_FORMAT=qwen3.5 USE_MTP=true bash rollout/sglang_server/run_turbo.sh
```

```bash
python rollout/single_turn_rollout.py --data-path data/eval/kernelbench/kernelbench.jsonl --save-path output/qwen3.5-turbo-test --model qwen3.5-35A3
```

## Eval Pipeline Return Values

The `run_eval_pipeline()` function returns a dictionary with the following top-level keys:

| Key | Type | Description |
|------|------|-------------|
| `ok` | bool | Whether the overall evaluation passed |
| `status` | str | Overall status: "passed" / "failed" / "warning" |
| `sample_id` | int | Sample ID |
| `started_at` | str | Start timestamp |
| `finished_at` | str | End timestamp |
| `duration_ms` | int | **Wall-clock execution time** (Python execution, not GPU kernel time) |
| `gates` | dict | Configured gates (require_compile_ok, etc.) |
| `gate_results` | dict | Gate check results (compile_gate, timing_gate, hack_gate) |
| `modules` | dict | Detailed results from each module (hack_check, compile, timing) |
| `summary` | dict | Aggregated summary |

### Module Results (each in `modules`)

Each module (hack_check, compile, timing) returns:

| Key | Type | Description |
|------|------|-------------|
| `module` | str | Module name |
| `ok` | bool | Whether this module passed |
| `status` | str | Module status |
| `duration_ms` | int | **Wall-clock execution time for this module** (Python execution time, not GPU kernel time) |
| `input` | dict | Input provided to the module |
| `metrics` | dict | Metrics/measurements from the module |
| `artifacts` | dict | Output artifacts from the module |
| `issues` | list | List of detected issues |
| `errors` | list | List of execution errors |

### Summary Fields (in `summary`)

| Key | Type | Description |
|------|------|-------------|
| `issues_total` | int | Total number of issues detected |
| `errors_total` | int | Total number of errors |
| `module_status` | dict | Status of each module |
| `hack_check.has_issue` | bool | Whether hack check found any issues |
| `hack_check.issue_total` | int | Total number of hack check issues |
| `hack_check.torch_reference_count` | int | Number of `torch.` references in code |
| `hack_check.torch_import_count` | int | Number of `import torch` statements |
| `hack_check.torch_symbol_total` | int | Total torch-related symbols |
| `hack_check.suspicious_torch_op_count` | int | Number of suspicious operations (matmul, mm, bmm, etc.) |
| `compile.return_code` | int | Compilation return code |
| `compile.error_category` | str | Category of compilation error |
| `timing.mean_ms` | float | **GPU kernel execution time** (from CUDA Graph timing) |
| `timing.median_ms` | float | **GPU kernel median time** (from CUDA Graph timing) |
| `timing.p10_ms` | float | 10th percentile of GPU kernel time |
| `timing.p90_ms` | float | 90th percentile of GPU kernel time |
| `timing.stdev_ms` | float | Standard deviation of GPU kernel time |

**Important:**
- `duration_ms` (at module level) = wall-clock time for Python execution
- `timing.mean_ms` (in summary) = GPU kernel execution time measured by CUDA Graph events

### Hack Check Metrics

The `metrics` field in hack_check module contains:

| Key | Description |
|------|-------------|
| `has_issue` | **yes** (true) if any issues found, **no** (false) if clean |
| `issue_total` | Total count of issues |
| `torch_reference_count` | Count of `torch.` in code |
| `torch_import_count` | Count of `import torch` statements |
| `torch_symbol_total` | torch_reference_count + torch_import_count |
| `suspicious_torch_op_count` | Number of suspicious operations detected |
| `issue_by_category` | Dict of issue counts by category |
| `torch_usage_policy` | Configured policy (allow/warn/forbid) |
| `precision_target` | Expected precision (fp32/fp16/bf16) |

### Timing Module Metrics (CUDA Graph)

The `metrics` field in timing module contains (when using cudagraph mode):

**Base metrics:**

| Key | Type | Description |
|------|------|-------------|
| `mode` | str | "cudagraph" - measurement mode |
| `device` | str | GPU device used |
| `success_rate` | float | Percentage of successful runs |
| `mean_ms` | float | Mean GPU kernel execution time |
| `median_ms` | float | Median GPU kernel execution time |
| `p10_ms` | float | 10th percentile |
| `p90_ms` | float | 90th percentile |
| `min_ms` | float | Minimum GPU kernel time |
| `max_ms` | float | Maximum GPU kernel time |
| `std_ms` | float | Standard deviation |
| `cv` | float | Coefficient of variation |
| `warmup_calls` | int | Number of warmup iterations |
| `graph_iters` | int | Number of iterations per CUDA graph |
| `num_replays` | int | Number of graph replays |
| `total_calls_per_sample` | int | Total kernel calls |
| `eager_probe_ms` | float | Time in eager (non-graph) mode |
| `suspicious` | bool | Whether timing is suspiciously low |

**Generated function metrics (with "generated_" prefix):**
- `generated_mean_ms`, `generated_median_ms`, `generated_p10_ms`, `generated_p90_ms`
- `generated_min_ms`, `generated_max_ms`, `generated_stdev_ms`
- `generated_cv` - coefficient of variation
- `generated_warmup_calls`, `generated_total_calls_per_sample`
- `generated_graph_iters`, `generated_num_replays`
- `generated_eager_probe_ms`
- `generated_suspicious`

**Reference function metrics (with "reference_" prefix, if provided):**
- Same fields as generated but with "reference_" prefix
- Only present if reference code was provided

### Issue Objects

Each issue in `issues` list contains:

| Key | Type | Description |
|------|------|-------------|
| `id` | str | Issue identifier (fabricated_output, passthrough_return, etc.) |
| `category` | str | Category (fake_optimization, torch_usage, etc.) |
| `message` | str | Human-readable message |
| `line` | int | Line number where issue detected |
| `span` | list | [start_pos, end_pos] in source |
| `snippet` | str | Code snippet around the issue |
| `details` | dict | Optional additional details |

### Return Status Values

| Status | Meaning |
|--------|---------|
| `ok` | No issues found, everything passed |
| `failed` | Issues detected in hack check |
| `skipped` | Module was skipped |
| `error` | Unexpected error during execution |
| `warning` | Issues found but not critical |

### Timing Details

**Two types of timing measurements exist:**

1. **`duration_ms`** (module-level and top-level)
   - Wall-clock time measured by Python `perf_counter()`
   - Includes Python code execution, function calls, data preparation
   - Measured from `started_timer()` to `finalize_module_result()`

2. **`timing.mean_ms`, `timing.median_ms`, etc.** (in summary for timing module)
   - GPU kernel execution time measured by CUDA Graph events (`torch.cuda.Event`)
   - Only available in timing module when using cudagraph mode
   - Represents actual GPU kernel time, excluding host-side overhead

