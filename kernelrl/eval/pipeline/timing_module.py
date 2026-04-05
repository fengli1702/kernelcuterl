"""
Timing and benchmarking module for CUDA kernel evaluation.

Supports two modes:
- run_cmd: Execute shell commands and measure wall-time performance
- cudagraph: Use CUDA graphs for in-process callable benchmarking
"""

from __future__ import annotations

import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .protocol import (
    add_error,
    add_issue,
    finalize_module_result,
    init_module_result,
    started_timer,
    truncate_text,
)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return -1.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo))


def _trimmed_mean(values: list[float], trim_ratio: float = 0.1) -> float:
    if not values:
        return -1.0
    n = len(values)
    k = int(n * trim_ratio)
    if n <= 2 * k:
        return float(statistics.mean(values))
    core = sorted(values)[k : n - k]
    return float(statistics.mean(core))


def _performance_summary(success_durations: list[float], baseline_ms: float | None) -> dict[str, Any]:
    if not success_durations:
        return {
            "baseline_ms": baseline_ms,
            "mean_ms": -1.0,
            "speedup_vs_baseline": None,
            "delta_ms_vs_baseline": None,
            "verdict": "no_successful_runs",
        }

    mean_ms = float(statistics.mean(success_durations))
    if baseline_ms is None or baseline_ms <= 0:
        return {
            "baseline_ms": baseline_ms,
            "mean_ms": round(mean_ms, 6),
            "speedup_vs_baseline": None,
            "delta_ms_vs_baseline": None,
            "verdict": "no_baseline",
        }

    speedup = baseline_ms / mean_ms if mean_ms > 0 else None
    delta = mean_ms - baseline_ms
    if speedup is None:
        verdict = "invalid_mean"
    elif speedup >= 1.05:
        verdict = "faster"
    elif speedup <= 0.95:
        verdict = "slower"
    else:
        verdict = "similar"

    return {
        "baseline_ms": round(float(baseline_ms), 6),
        "mean_ms": round(mean_ms, 6),
        "speedup_vs_baseline": round(float(speedup), 6) if speedup is not None else None,
        "delta_ms_vs_baseline": round(float(delta), 6),
        "verdict": verdict,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    sv = sorted(values)
    pos = (len(sv) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sv[lo])
    w = pos - lo
    return float((1.0 - w) * sv[lo] + w * sv[hi])


@dataclass
class CUDAGraphTimingResult:
    samples_ms: list[float]
    warmup_calls: int
    total_calls_per_sample: int
    graph_iters: int
    num_replays: int
    eager_probe_ms: float | None
    suspicious: bool
    suspicious_reason: str | None

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.samples_ms))

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @property
    def stdev_ms(self) -> float:
        if len(self.samples_ms) <= 1:
            return 0.0
        return float(statistics.stdev(self.samples_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.samples_ms))

    @property
    def max_ms(self) -> float:
        return float(max(self.samples_ms))

    @property
    def p10_ms(self) -> float:
        return _quantile(self.samples_ms, 0.10)

    @property
    def p90_ms(self) -> float:
        return _quantile(self.samples_ms, 0.90)

    @property
    def cv(self) -> float:
        mean = self.mean_ms
        if mean == 0.0:
            return 0.0
        return float(self.stdev_ms / mean)


def _event_probe_ms(
    fn: Callable[[], None],
    *,
    device: "Any",
    probe_calls: int,
) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    start.record()
    for _ in range(max(1, probe_calls)):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, probe_calls))


def benchmark_with_cudagraph(
    fn: Callable[..., Any],
    *,
    device: "Any",
    fn_args: Sequence[Any] = (),
    fn_kwargs: Mapping[str, Any] | None = None,
    warmup: int = 10,
    warmup_ms: float | None = None,
    graph_iters: int = 10,
    pre_capture_iters: int = 3,
    trial_count: int = 9,
    min_measure_ms: float = 200.0,
    min_replays: int = 5,
    max_replays: int = 200000,
    fixed_repeat_calls: int | None = None,
    use_default_stream: bool = True,
    setup_fn: Callable[[], None] | None = None,
    probe_calls: int = 20,
    suspicious_ratio_threshold: float = 0.25,
    allow_suspicious: bool = True,
) -> CUDAGraphTimingResult:
    import torch

    call_kwargs = dict(fn_kwargs or {})

    def _call_once() -> None:
        fn(*fn_args, **call_kwargs)

    if setup_fn is not None:
        setup_fn()

    warmup_calls = max(1, warmup)
    for _ in range(warmup_calls):
        _call_once()

    if warmup_ms is not None and warmup_ms > 0.0:
        est_ms = _event_probe_ms(_call_once, device=device, probe_calls=5)
        if est_ms > 0.0:
            target_calls = int(math.ceil(float(warmup_ms) / est_ms))
            target_calls = max(target_calls, 1)
            extra_calls = max(0, target_calls - warmup_calls)
            for _ in range(extra_calls):
                _call_once()
            warmup_calls += extra_calls
    torch.cuda.synchronize(device=device)

    graph = torch.cuda.CUDAGraph()
    graph_stream = (
        torch.cuda.default_stream(device=device)
        if use_default_stream
        else torch.cuda.Stream(device=device)
    )
    caller_stream = torch.cuda.current_stream(device=device)
    graph_stream.wait_stream(caller_stream)
    with torch.cuda.stream(graph_stream):
        for _ in range(max(0, pre_capture_iters)):
            _call_once()
        with torch.cuda.graph(graph):
            for _ in range(max(1, graph_iters)):
                _call_once()
    caller_stream.wait_stream(graph_stream)
    torch.cuda.synchronize(device=device)

    estimate_replays = 5
    est_start = torch.cuda.Event(enable_timing=True)
    est_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(graph_stream):
        est_start.record()
        for _ in range(estimate_replays):
            graph.replay()
        est_end.record()
    caller_stream.wait_stream(graph_stream)
    torch.cuda.synchronize(device=device)
    est_replay_ms = float(est_start.elapsed_time(est_end)) / float(estimate_replays)

    if fixed_repeat_calls is not None:
        num_replays = int(math.ceil(float(max(1, fixed_repeat_calls)) / float(max(1, graph_iters))))
    else:
        if est_replay_ms <= 0.0:
            num_replays = max(1, min_replays)
        else:
            num_replays = int(math.ceil(float(max(1.0, min_measure_ms)) / est_replay_ms))
            num_replays = max(num_replays, max(1, min_replays))
    num_replays = min(num_replays, max(1, max_replays))
    total_calls = max(1, num_replays * max(1, graph_iters))

    samples: list[float] = []
    for _ in range(max(1, trial_count)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start.record()
            for _ in range(num_replays):
                graph.replay()
            end.record()
        caller_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(device=device)
        total_ms = float(start.elapsed_time(end))
        samples.append(total_ms / float(total_calls))

    eager_probe_ms = _event_probe_ms(_call_once, device=device, probe_calls=probe_calls)
    median_ms = float(statistics.median(samples))
    suspicious = False
    suspicious_reason: str | None = None
    if eager_probe_ms > 0.0 and median_ms < suspicious_ratio_threshold * eager_probe_ms:
        suspicious = True
        suspicious_reason = (
            f"graph median {median_ms:.6f} ms is < "
            f"{suspicious_ratio_threshold:.2f}x eager probe {eager_probe_ms:.6f} ms"
        )
        if not allow_suspicious:
            raise RuntimeError(f"suspicious CUDA graph timing: {suspicious_reason}")

    return CUDAGraphTimingResult(
        samples_ms=samples,
        warmup_calls=warmup_calls,
        total_calls_per_sample=total_calls,
        graph_iters=max(1, graph_iters),
        num_replays=num_replays,
        eager_probe_ms=eager_probe_ms,
        suspicious=suspicious,
        suspicious_reason=suspicious_reason,
    )


def time_with_cudagraph(
    fn: Callable[..., Any],
    *,
    device: "Any",
    fn_args: Sequence[Any] = (),
    fn_kwargs: Mapping[str, Any] | None = None,
    warmup: int = 10,
    warmup_ms: float | None = None,
    repeat: int = 50,
    graph_iters: int = 10,
    pre_capture_iters: int = 3,
    use_default_stream: bool = True,
    setup_fn: Callable[[], None] | None = None,
) -> float:
    result = benchmark_with_cudagraph(
        fn=fn,
        device=device,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        warmup=warmup,
        warmup_ms=warmup_ms,
        graph_iters=graph_iters,
        pre_capture_iters=pre_capture_iters,
        trial_count=1,
        fixed_repeat_calls=repeat,
        use_default_stream=use_default_stream,
        setup_fn=setup_fn,
        allow_suspicious=True,
    )
    return result.median_ms


def _run_timing_module_run_cmd(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    start = started_timer()

    run_cmd = request.get("run_cmd")
    if not run_cmd:
        add_issue(result, "timing_skipped", "medium", "run_cmd is missing")
        return finalize_module_result(result, start, ok=False, status="skipped")

    try:
        warmup = max(0, int(request.get("warmup", 1)))
        repeats = max(1, int(request.get("repeats", 5)))
        timeout_sec = int(request.get("timeout_sec", 120))
        shell = bool(request.get("shell", True))
        require_success_all = bool(request.get("require_success_all", True))
        capture_output_chars = int(request.get("capture_output_chars", 2000))

        baseline_ms_val = request.get("baseline_ms")
        baseline_ms = float(baseline_ms_val) if baseline_ms_val is not None else None

        env = os.environ.copy()
        user_env = request.get("env") or {}
        if isinstance(user_env, dict):
            env.update({str(k): str(v) for k, v in user_env.items()})

        cwd = request.get("cwd")
        total_runs = warmup + repeats

        successful_measured = 0
        failed_measured = 0
        timeout_measured = 0

        measured_durations_ms: list[float] = []
        measured_runs: list[dict[str, Any]] = []
        warmup_runs: list[dict[str, Any]] = []

        for i in range(total_runs):
            is_warmup = i < warmup
            run_start = perf_counter()

            try:
                proc = subprocess.run(
                    run_cmd,
                    shell=shell,
                    cwd=cwd,
                    env=env,
                    timeout=timeout_sec,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                wall_ms = (perf_counter() - run_start) * 1000.0
                run_row = {
                    "index": i,
                    "phase": "warmup" if is_warmup else "measure",
                    "return_code": proc.returncode,
                    "wall_time_ms": round(wall_ms, 3),
                    "stdout": truncate_text(proc.stdout, max_chars=capture_output_chars),
                    "stderr": truncate_text(proc.stderr, max_chars=capture_output_chars),
                }

                if is_warmup:
                    warmup_runs.append(run_row)
                    continue

                measured_runs.append(run_row)
                if proc.returncode == 0:
                    measured_durations_ms.append(wall_ms)
                    successful_measured += 1
                else:
                    failed_measured += 1
                    add_issue(
                        result,
                        "timing_run_failed",
                        "medium",
                        "One measured timing run failed",
                        details={"index": i, "return_code": proc.returncode},
                    )
            except subprocess.TimeoutExpired:
                wall_ms = (perf_counter() - run_start) * 1000.0
                run_row = {
                    "index": i,
                    "phase": "warmup" if is_warmup else "measure",
                    "return_code": None,
                    "timeout": True,
                    "wall_time_ms": round(wall_ms, 3),
                }
                if is_warmup:
                    warmup_runs.append(run_row)
                else:
                    measured_runs.append(run_row)
                    timeout_measured += 1
                    add_issue(
                        result,
                        "timing_run_timeout",
                        "high",
                        "One measured timing run timed out",
                        details={"index": i, "timeout_sec": timeout_sec},
                    )

        sorted_durations = sorted(measured_durations_ms)
        mean_ms = float(statistics.mean(sorted_durations)) if sorted_durations else -1.0
        std_ms = float(statistics.pstdev(sorted_durations)) if len(sorted_durations) > 1 else 0.0
        cv = (std_ms / mean_ms) if mean_ms > 0 else None

        stats = {
            "mode": "run_cmd",
            "warmup": warmup,
            "repeats": repeats,
            "successful_measured_runs": successful_measured,
            "failed_measured_runs": failed_measured,
            "timeout_measured_runs": timeout_measured,
            "success_rate": round(successful_measured / repeats, 4),
            "mean_ms": round(mean_ms, 3) if mean_ms >= 0 else -1.0,
            "trimmed_mean_ms": round(_trimmed_mean(sorted_durations, trim_ratio=0.1), 3) if sorted_durations else -1.0,
            "std_ms": round(std_ms, 3),
            "cv": round(cv, 6) if cv is not None else None,
            "min_ms": round(min(sorted_durations), 3) if sorted_durations else -1.0,
            "max_ms": round(max(sorted_durations), 3) if sorted_durations else -1.0,
            "p50_ms": round(_percentile(sorted_durations, 0.50), 3) if sorted_durations else -1.0,
            "p90_ms": round(_percentile(sorted_durations, 0.90), 3) if sorted_durations else -1.0,
            "p95_ms": round(_percentile(sorted_durations, 0.95), 3) if sorted_durations else -1.0,
        }

        perf = _performance_summary(sorted_durations, baseline_ms)

        failures = [x for x in measured_runs if x.get("return_code") not in (0, None) or x.get("timeout")]

        result["metrics"].update(stats)
        result["artifacts"].update(
            {
                "run_cmd": run_cmd,
                "cwd": cwd,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "failure_examples": failures[:10],
                "performance": perf,
            }
        )

        if successful_measured == 0:
            add_error(result, "no_successful_timing_runs", "No successful measured run")
            return finalize_module_result(result, start, ok=False, status="error")

        if require_success_all and successful_measured != repeats:
            add_error(
                result,
                "timing_partial_success",
                "Not all measured timing runs were successful",
                details={"successful": successful_measured, "expected": repeats},
            )
            return finalize_module_result(result, start, ok=False, status="partial_success")

        if successful_measured != repeats:
            return finalize_module_result(result, start, ok=True, status="partial_success")

        return finalize_module_result(result, start, ok=True, status="ok")

    except Exception as exc:  # pragma: no cover - defensive
        add_error(result, "unexpected_exception", str(exc))
        return finalize_module_result(result, start, ok=False, status="error")


def _cudagraph_metrics(name: str, timing: CUDAGraphTimingResult) -> dict[str, Any]:
    return {
        f"{name}_mean_ms": round(timing.mean_ms, 6),
        f"{name}_median_ms": round(timing.median_ms, 6),
        f"{name}_stdev_ms": round(timing.stdev_ms, 6),
        f"{name}_min_ms": round(timing.min_ms, 6),
        f"{name}_max_ms": round(timing.max_ms, 6),
        f"{name}_p10_ms": round(timing.p10_ms, 6),
        f"{name}_p90_ms": round(timing.p90_ms, 6),
        f"{name}_cv": round(timing.cv, 8),
        f"{name}_warmup_calls": int(timing.warmup_calls),
        f"{name}_total_calls_per_sample": int(timing.total_calls_per_sample),
        f"{name}_graph_iters": int(timing.graph_iters),
        f"{name}_num_replays": int(timing.num_replays),
        f"{name}_eager_probe_ms": round(float(timing.eager_probe_ms), 6) if timing.eager_probe_ms is not None else None,
        f"{name}_suspicious": bool(timing.suspicious),
    }


def _run_timing_module_cudagraph(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    start = started_timer()

    try:
        import torch
    except Exception as exc:
        add_error(result, "torch_unavailable", f"PyTorch is required for cudagraph mode: {exc}")
        return finalize_module_result(result, start, ok=False, status="error")

    generated_fn = request.get("generated_fn") or request.get("fn")
    reference_fn = request.get("reference_fn")

    if not callable(generated_fn):
        add_error(result, "generated_fn_missing", "generated_fn (or fn) must be callable in cudagraph mode")
        return finalize_module_result(result, start, ok=False, status="error")

    device_req = request.get("device", "cuda")
    device = torch.device(device_req)
    if device.type == "cuda" and not torch.cuda.is_available():
        add_error(result, "cuda_unavailable", "CUDA device requested but torch.cuda.is_available() is false")
        return finalize_module_result(result, start, ok=False, status="error")

    if device.type == "cuda":
        torch.cuda.set_device(device)

    generated_args = tuple(request.get("generated_fn_args", request.get("fn_args", ())))
    generated_kwargs = dict(request.get("generated_fn_kwargs", request.get("fn_kwargs", {})) or {})
    generated_setup_fn = request.get("generated_setup_fn", request.get("setup_fn"))

    reference_args = tuple(request.get("reference_fn_args", generated_args))
    reference_kwargs = dict(request.get("reference_fn_kwargs", generated_kwargs) or {})
    reference_setup_fn = request.get("reference_setup_fn", generated_setup_fn)

    warmup = int(request.get("warmup", 10))
    warmup_ms = request.get("warmup_ms")
    graph_iters = int(request.get("graph_iters", 10))
    pre_capture_iters = int(request.get("pre_capture_iters", 3))
    trial_count = int(request.get("trial_count", 9))
    min_measure_ms = float(request.get("min_measure_ms", 200.0))
    min_replays = int(request.get("min_replays", 5))
    max_replays = int(request.get("max_replays", 200000))
    fixed_repeat_calls = request.get("fixed_repeat_calls")
    if fixed_repeat_calls is not None:
        fixed_repeat_calls = int(fixed_repeat_calls)
    use_default_stream = bool(request.get("use_default_stream", True))
    probe_calls = int(request.get("probe_calls", 20))
    suspicious_ratio_threshold = float(request.get("suspicious_ratio_threshold", 0.25))
    allow_suspicious = bool(request.get("allow_suspicious", True))
    require_not_suspicious = bool(request.get("require_not_suspicious", False))

    generated_result = benchmark_with_cudagraph(
        fn=generated_fn,
        device=device,
        fn_args=generated_args,
        fn_kwargs=generated_kwargs,
        warmup=warmup,
        warmup_ms=warmup_ms,
        graph_iters=graph_iters,
        pre_capture_iters=pre_capture_iters,
        trial_count=trial_count,
        min_measure_ms=min_measure_ms,
        min_replays=min_replays,
        max_replays=max_replays,
        fixed_repeat_calls=fixed_repeat_calls,
        use_default_stream=use_default_stream,
        setup_fn=generated_setup_fn,
        probe_calls=probe_calls,
        suspicious_ratio_threshold=suspicious_ratio_threshold,
        allow_suspicious=allow_suspicious,
    )

    reference_result: CUDAGraphTimingResult | None = None
    if callable(reference_fn):
        reference_result = benchmark_with_cudagraph(
            fn=reference_fn,
            device=device,
            fn_args=reference_args,
            fn_kwargs=reference_kwargs,
            warmup=warmup,
            warmup_ms=warmup_ms,
            graph_iters=graph_iters,
            pre_capture_iters=pre_capture_iters,
            trial_count=trial_count,
            min_measure_ms=min_measure_ms,
            min_replays=min_replays,
            max_replays=max_replays,
            fixed_repeat_calls=fixed_repeat_calls,
            use_default_stream=use_default_stream,
            setup_fn=reference_setup_fn,
            probe_calls=probe_calls,
            suspicious_ratio_threshold=suspicious_ratio_threshold,
            allow_suspicious=allow_suspicious,
        )

    baseline_ms_val = request.get("baseline_ms")
    baseline_ms = float(baseline_ms_val) if baseline_ms_val is not None else None
    if baseline_ms is None and reference_result is not None:
        baseline_ms = reference_result.mean_ms

    perf = _performance_summary(generated_result.samples_ms, baseline_ms)

    metrics: dict[str, Any] = {
        "mode": "cudagraph",
        "device": str(device),
        "success_rate": 1.0,
        "mean_ms": round(generated_result.mean_ms, 6),
        "p90_ms": round(generated_result.p90_ms, 6),
        "median_ms": round(generated_result.median_ms, 6),
        "std_ms": round(generated_result.stdev_ms, 6),
        "cv": round(generated_result.cv, 8),
        "min_ms": round(generated_result.min_ms, 6),
        "max_ms": round(generated_result.max_ms, 6),
        "p10_ms": round(generated_result.p10_ms, 6),
        "warmup_calls": int(generated_result.warmup_calls),
        "graph_iters": int(generated_result.graph_iters),
        "num_replays": int(generated_result.num_replays),
        "total_calls_per_sample": int(generated_result.total_calls_per_sample),
        "eager_probe_ms": round(float(generated_result.eager_probe_ms), 6)
        if generated_result.eager_probe_ms is not None
        else None,
        "suspicious": bool(generated_result.suspicious),
    }
    metrics.update(_cudagraph_metrics("generated", generated_result))
    if reference_result is not None:
        metrics.update(_cudagraph_metrics("reference", reference_result))

    result["metrics"].update(metrics)
    result["artifacts"].update(
        {
            "performance": perf,
            "generated_timing": asdict(generated_result),
            "reference_timing": asdict(reference_result) if reference_result is not None else None,
            "request_echo": {
                "sample_id": request.get("sample_id"),
                "has_reference_code": isinstance(request.get("reference_code"), str),
                "has_generated_code": isinstance(request.get("generated_code"), str),
                "generated_code_truncated": request.get("generated_code_truncated"),
                "generated_code_truncation_reason": request.get("generated_code_truncation_reason"),
            },
        }
    )

    if generated_result.suspicious:
        add_issue(
            result,
            "cudagraph_suspicious_generated",
            "medium",
            "Generated function CUDA graph timing looks suspiciously low",
            details={"reason": generated_result.suspicious_reason},
        )

    if reference_result is not None and reference_result.suspicious:
        add_issue(
            result,
            "cudagraph_suspicious_reference",
            "medium",
            "Reference function CUDA graph timing looks suspiciously low",
            details={"reason": reference_result.suspicious_reason},
        )

    if require_not_suspicious and (generated_result.suspicious or (reference_result and reference_result.suspicious)):
        add_error(result, "cudagraph_suspicious", "Suspicious CUDA graph timing result")
        return finalize_module_result(result, start, ok=False, status="partial_success")

    return finalize_module_result(result, start, ok=True, status="ok")


def run_timing_module(request: dict[str, Any]) -> dict[str, Any]:
    """
    Timing module interface (dict in, dict out).

    Modes:
    - run_cmd (legacy): executes shell command repeatedly and reports wall-time stats.
    - cudagraph: uses in-process callable(s) and CUDA Graph timing utility.

    Mode selection:
    - request["mode"] if provided
    - else "cudagraph" when generated_fn/fn exists
    - else "run_cmd" when run_cmd exists
    - else skipped
    """
    result = init_module_result("timing", request)

    mode = request.get("mode")
    if not mode:
        if callable(request.get("generated_fn")) or callable(request.get("fn")):
            mode = "cudagraph"
        elif request.get("run_cmd"):
            mode = "run_cmd"
        else:
            mode = "skipped"

    mode = str(mode).strip().lower()
    result["artifacts"]["selected_mode"] = mode

    if mode == "cudagraph":
        return _run_timing_module_cudagraph(result, request)
    if mode == "run_cmd":
        return _run_timing_module_run_cmd(result, request)

    start = started_timer()
    add_issue(result, "timing_skipped", "medium", f"unsupported or missing timing mode: {mode}")
    return finalize_module_result(result, start, ok=False, status="skipped")
