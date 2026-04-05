# Eval Pipeline Interface

This document defines the new dictionary-based interfaces under `src/eval_pipeline/`.

## 1. Compile Module

Function: `run_compile_module(request: dict) -> dict`

Request (recommended keys):
- `language`: `python|cuda|cpp|c++|c`
- `code`: source code string (optional)
- `source_path`: existing source file path (optional)
- `build_dir`: build workspace path (optional)
- `filename`: filename when `code` is provided (optional)
- `compile_cmd`: explicit compile command (optional)
- `timeout_sec`: compile timeout, default `60`
- `env`: env overrides map (optional)
- `shell`: whether to run command in shell, default `true`
- `capture_output_chars`: truncate stdout/stderr to this size
- `collect_build_outputs`: include generated files metadata, default `true`

Return:
- Always returns a dict with keys:
  - `module`, `ok`, `status`, `started_at`, `finished_at`, `duration_ms`
  - `input`, `metrics`, `artifacts`, `issues`, `errors`
- `metrics` adds:
  - `return_code`, `warnings_count`, `errors_count`, `error_category`
  - `compiled_artifact_present`, `build_output_count`
- `artifacts` adds:
  - `diagnostics.warnings[]`, `diagnostics.errors[]` (line/text)
  - `build_outputs[]` (relative file path + size)

## 2. Timing Module

Function: `run_timing_module(request: dict) -> dict`

Request supports two modes:

1) `mode = "run_cmd"` (legacy)
- `run_cmd`: command to benchmark
- `cwd`: working dir (optional)
- `env`: env overrides map (optional)
- `warmup`: warmup runs, default `1`
- `repeats`: measured runs, default `5`
- `timeout_sec`: per-run timeout, default `120`
- `shell`: shell mode, default `true`
- `require_success_all`: fail module if any measured run fails, default `true`
- `capture_output_chars`: truncate stdout/stderr size
- `baseline_ms`: optional baseline latency for speedup verdict

2) `mode = "cudagraph"` (new)
- `generated_fn` (or `fn`): callable to benchmark (required)
- `reference_fn`: optional callable baseline
- `device`: torch device string, default `cuda`
- `generated_fn_args/generated_fn_kwargs` and `reference_fn_args/reference_fn_kwargs` (optional)
- `generated_setup_fn/reference_setup_fn` (optional)
- timing controls:
  - `warmup` (default `10`)
  - `warmup_ms` (optional)
  - `graph_iters` (default `10`)
  - `pre_capture_iters` (default `3`)
  - `trial_count` (default `9`)
  - `min_measure_ms` (default `200.0`)
  - `min_replays` (default `5`)
  - `max_replays` (default `200000`)
  - `fixed_repeat_calls` (optional)
  - `use_default_stream` (default `true`)
  - `probe_calls` (default `20`)
  - `suspicious_ratio_threshold` (default `0.25`)
  - `allow_suspicious` (default `true`)
  - `require_not_suspicious` (default `false`)

Return metrics include:
- `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `p50_ms`, `p90_ms`
- `trimmed_mean_ms`, `p95_ms`, `success_rate`, `cv`
- run counts: `successful_measured_runs`, `failed_measured_runs`, `timeout_measured_runs`
- `artifacts.performance`:
  - `baseline_ms`, `mean_ms`, `speedup_vs_baseline`, `delta_ms_vs_baseline`, `verdict`
- `artifacts.failure_examples[]` includes failed/timeout run snapshots

For `cudagraph` mode, metrics additionally include:
- `median_ms`, `p10_ms`, `warmup_calls`, `graph_iters`, `num_replays`
- `total_calls_per_sample`, `eager_probe_ms`, `suspicious`
- prefixed sets for both sides when `reference_fn` exists:
  - `generated_*`, `reference_*`

Mode auto-selection:
- if `mode` missing and `generated_fn/fn` is callable -> selects `cudagraph`
- else if `run_cmd` exists -> selects `run_cmd`
- else returns `skipped`

## 3. Hack Check Module

Function: `run_hack_check_module(request: dict) -> dict`

Request (recommended keys):
- `code`: code to check (preferred)
- `response_text`: fallback text payload to check
- `strict`: if true, medium severity issues fail the module
- `torch_usage_policy`: `allow|warn|forbid`, default `warn`
- `precision_target`: `fp32|fp16|bf16` (for precision downgrade checks)
- `expected_backend`: `cuda|triton|hip`
- `expected_custom_signals`: extra regex list
- `min_custom_signal_hits`: default `1`

Checks currently include:
- dangerous exec/eval usage
- shell/process invocation
- network calls
- torch fallback patterns
- silent exception swallowing
- hard-coded success markers
- fabricated outputs / passthrough returns
- precision downgrade under fp32 target
- missing backend custom-kernel signals
- suspicious torch op usage (`torch.mm`, `torch.nn.functional.*`, etc.)

Return metrics include:
- `checked_chars`, `issue_total`, `high`, `medium`, `low`, `strict`
- `issue_by_category`
- `artifacts.torch_usage` (`torch_reference_count`, `torch_import_count`, `torch_symbol_total`, suspicious hits)
- `artifacts.custom_signal` (pattern list + hit details)

## 4. Unified Orchestrator

Function: `run_eval_pipeline(request: dict) -> dict`

Request:
- `sample_id` (optional)
- `reference_code` (optional)
- `generated_code` (optional)
- `generated_code_truncated` (optional)
- `use_reference_on_generated_truncation` (default `true`)
- `generated_code_min_len` (default `48`)
- `compile`: compile module request
- `timing`: timing module request
- `hack_check`: hack-check module request
- `gates` (optional):
  - `require_compile_ok` (default `true`)
  - `require_timing_ok` (default `false`)
  - `require_hack_clean` (default `true`)

Return:
- top-level verdict: `ok`, `status`
- `gate_results` for each gate
- module outputs in `modules.{hack_check,compile,timing}`
- aggregate counters in `summary`
- `summary.code_selection`:
  - `generated_code_truncated`
  - `generated_code_truncation_reason`
  - `used_reference_as_generated`
  - `used_reference_fn_as_generated`
- `summary` adds compact fields for:
  - compile: `return_code/error_category/warnings_count/errors_count`
  - timing: `mean_ms/p90_ms/success_rate/performance_verdict/speedup_vs_baseline`
  - hack_check: severity counts + torch usage counters

## 5. Demo

Run:

```bash
python3 scripts/eval_pipeline_demo.py
```

This prints one full end-to-end output dict that can be directly consumed by higher-level systems (including external `kernelrl` wrappers).

## 6. Truncation Fallback Behavior

If `generated_code` is missing/empty/too-short/contains truncation markers (or `generated_code_truncated=true`) and `use_reference_on_generated_truncation=true`:
- pipeline sets `effective_generated_code = reference_code`
- pipeline forwards `reference_code`, `generated_code`, and `effective_generated_code` into both compile/timing module inputs
- compile module uses `effective_generated_code` when `compile.code` is not explicitly provided
- when timing uses callable mode and `timing.reference_fn` exists, pipeline also maps `timing.generated_fn = timing.reference_fn` by default (`use_reference_fn_on_generated_truncation=true`)
