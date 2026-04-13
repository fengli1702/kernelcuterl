# KernelRL Compilation & Timing Modules Report

## 1. COMPILATION MODULE - Input/Output Interface

### 1.1 Compilation Input (run_compile_module request)

```python
{
    # Required
    "language": "python",  # or "cuda", "cpp", "c"

    # Code source (provide ONE of these)
    "code": "def add(a, b):\n    return a + b",  # Source code string
    # OR
    "source_path": "/path/to/source.py",  # Existing file path

    # Optional configuration
    "build_dir": "/tmp/build_12345",  # Build directory (auto-created if not provided)
    "filename": "candidate.py",  # Output filename (default: candidate.py/candidate.cu/candidate.cpp)
    "compile_cmd": "/usr/bin/python3 -m py_compile ...",  # Custom compile command
    "timeout_sec": 60,  # Compilation timeout (default: 60)
    "env": {  # Environment variables (optional)
        "CUDA_VISIBLE_DEVICES": "0",
        "PATH": "/usr/local/cuda/bin:..."
    },
    "shell": True,  # Execute as shell command (default: True)
    "capture_output_chars": 4000,  # Max stderr/stdout capture (default: 4000)
    "collect_build_outputs": True  # Collect artifacts (default: True)
}
```

### 1.2 Compilation Output (Result Dictionary)

```python
{
    "module": "compile",
    "ok": True,  # Success/failure
    "status": "ok",  # "ok", "skipped", or "error"
    "started_at": "2026-04-05T14:27:41.869216+00:00",
    "finished_at": "2026-04-05T14:27:41.922148+00:00",
    "duration_ms": 52,  # Python execution time (wall-clock)

    # Core metrics
    "metrics": {
        "return_code": 0,  # 0 = success, non-zero = failure
        "error_category": "none",  # Error type if return_code != 0
        "warnings_count": 0,
        "errors_count": 0,
        "timeout_sec": 60,
        "build_output_count": 2,  # Number of artifacts generated
        "compiled_artifact_present": False  # .o, .so, .pyd, .dll present?
    },

    # Detailed outputs
    "artifacts": {
        "source_path": "/tmp/kernel_eval_build_xyz/candidate.py",
        "build_dir": "/tmp/kernel_eval_build_xyz",
        "compile_cmd": "/usr/bin/python3 -m py_compile /tmp/.../candidate.py",
        "stdout": "",  # Captured standard output
        "stderr": "",  # Captured error output
        "diagnostics": {
            "warnings": [
                {"line_no": 5, "text": "warning: unused variable 'x'"},
                {"line_no": 10, "text": "warning: deprecated function"}
            ],
            "errors": [
                {"line_no": 15, "text": "error: syntax error near line 15"}
            ]
        },
        "build_outputs": [
            {"path": "__pycache__/candidate.cpython-312.pyc", "size_bytes": 1752},
            {"path": "candidate.py", "size_bytes": 792}
        ]
    },

    "issues": [  # Detected problems
        {
            "id": "compile_warning",
            "severity": "low",
            "message": "Compilation succeeded with warnings",
            "details": {"warnings_count": 2}
        }
    ],

    "errors": []  # Execution errors
}
```

### 1.3 Compilation Error Categories

| error_category | Meaning | Example |
|----------------|---------|---------|
| `none` | No error (return_code = 0) | Successful compilation |
| `syntax_error` | Syntax error in code | Missing semicolon, invalid syntax |
| `missing_include_or_source` | File not found | #include <missing.h> |
| `link_error` | Linker error | undefined reference to function |
| `tool_not_found_or_not_executable` | Compiler not found (exit code 127) | nvcc not in PATH |
| `permission_error` | File permission denied | Cannot write to build dir |
| `oom` | Out of memory | Compilation ran out of RAM |
| `compile_error` | Generic compilation error | Unknown error |

---

## 2. TIMING MODULE - Input/Output Interface

### 2.1 Timing Input (run_timing_module request)

```python
{
    # Required: Function to benchmark (must be callable)
    "generated_fn": my_forward_function,  # Callable: def forward(A, B): return ...

    # Function arguments
    "generated_fn_args": [],  # Positional args: [arg1, arg2]
    "generated_fn_kwargs": {},  # Keyword args: {"param": value}

    # Optional: Reference function for comparison/speedup calculation
    "reference_fn": baseline_forward_function,  # Callable
    "reference_fn_args": [],
    "reference_fn_kwargs": {},

    # Setup functions (called before each measurement iteration)
    "generated_setup_fn": setup_generated,  # Optional: def setup(): prepare data
    "reference_setup_fn": setup_reference,  # Optional

    # Timing configuration
    "baseline_ms": 100.5,  # Explicit baseline time (if no reference_fn)
    "warmup_iters": 3,  # Warmup iterations (default: 3)
    "graph_iters": 10,  # Iterations per CUDA graph (default: 10)
    "num_replays": 5,  # Number of times to replay graph (default: 5)

    # GPU configuration
    "device": "cuda:0",  # GPU device (default: "cuda:0")

    # Source code (for context)
    "reference_code": "def forward(A, B):\n    return torch.matmul(A, B)",
    "generated_code": "def forward(A, B):\n    return torch.matmul(A, B)"
}
```

### 2.2 Timing Output (Result Dictionary)

```python
{
    "module": "timing",
    "ok": True,
    "status": "ok",  # "ok", "skipped", or "error"
    "started_at": "2026-04-05T14:27:41.922127+00:00",
    "finished_at": "2026-04-05T14:27:41.980000+00:00",
    "duration_ms": 52,  # Python execution time (wall-clock, NOT GPU kernel time)

    # Comprehensive timing metrics (GPU kernel times via CUDA events)
    "metrics": {
        # Meta information
        "mode": "cudagraph",
        "device": "cuda:0",
        "success_rate": 1.0,

        # Generated function timing (ACTUAL GPU KERNEL TIME)
        "mean_ms": 12.345,       # Average GPU kernel execution time
        "median_ms": 12.340,     # Median (p50)
        "p10_ms": 12.100,        # 10th percentile
        "p90_ms": 12.600,        # 90th percentile
        "min_ms": 12.050,        # Minimum
        "max_ms": 12.750,        # Maximum
        "std_ms": 0.245,         # Standard deviation
        "cv": 0.0199,            # Coefficient of variation

        # CUDA Graph parameters
        "warmup_calls": 3,
        "graph_iters": 10,
        "num_replays": 5,
        "total_calls_per_sample": 50,  # graph_iters * num_replays
        "eager_probe_ms": 15.600,  # Non-graph mode execution time
        "suspicious": False,  # Unrealistically low timing?

        # Generated function details (with "generated_" prefix)
        "generated_mean_ms": 12.345,
        "generated_median_ms": 12.340,
        "generated_p10_ms": 12.100,
        "generated_p90_ms": 12.600,
        "generated_min_ms": 12.050,
        "generated_max_ms": 12.750,
        "generated_stdev_ms": 0.245,
        "generated_cv": 0.0199,
        "generated_warmup_calls": 3,
        "generated_total_calls_per_sample": 50,
        "generated_graph_iters": 10,
        "generated_num_replays": 5,
        "generated_eager_probe_ms": 15.600,
        "generated_suspicious": False,

        # Reference function details (with "reference_" prefix, if provided)
        "reference_mean_ms": 15.678,
        "reference_median_ms": 15.670,
        "reference_p10_ms": 15.400,
        "reference_p90_ms": 15.950,
        "reference_min_ms": 15.200,
        "reference_max_ms": 16.100,
        "reference_stdev_ms": 0.280,
        "reference_cv": 0.0179,
        "reference_warmup_calls": 3,
        "reference_total_calls_per_sample": 50,
        "reference_graph_iters": 10,
        "reference_num_replays": 5,
        "reference_eager_probe_ms": 18.900,
        "reference_suspicious": False
    },

    # Performance comparison artifacts
    "artifacts": {
        "performance": {
            "baseline_ms": 15.678,  # Reference time or explicit baseline
            "mean_ms": 12.345,  # Generated function mean
            "speedup_vs_baseline": 1.269,  # 15.678 / 12.345 = 1.269x faster
            "delta_ms_vs_baseline": -3.333,  # 12.345 - 15.678
            "verdict": "faster"  # "faster", "slower", "similar", "no_baseline", "no_successful_runs"
        },

        # Raw timing result objects
        "generated_timing": {
            "mean_ms": 12.345,
            "median_ms": 12.340,
            "p10_ms": 12.100,
            "p90_ms": 12.600,
            "min_ms": 12.050,
            "max_ms": 12.750,
            "stdev_ms": 0.245,
            "cv": 0.0199,
            "warmup_calls": 3,
            "total_calls_per_sample": 50,
            "graph_iters": 10,
            "num_replays": 5,
            "eager_probe_ms": 15.600,
            "suspicious": False,
            "samples_ms": [12.345, 12.340, 12.350, ...]  # Raw samples
        },

        "reference_timing": {  # If reference function provided
            "mean_ms": 15.678,
            "median_ms": 15.670,
            # ... same structure as generated_timing
        }
    },

    "issues": [],
    "errors": []
}
```

### 2.3 Timing Measurements Explanation

| Metric | Unit | Measured By | What It Means |
|--------|------|-------------|---------------|
| `duration_ms` | ms | Python timer (perf_counter) | Wall-clock time for Python execution |
| `mean_ms` | ms | CUDA Event | Average GPU kernel execution time |
| `median_ms` | ms | CUDA Event | Middle value of GPU times |
| `p10_ms` | ms | CUDA Event | 10% of runs were faster than this |
| `p90_ms` | ms | CUDA Event | 90% of runs were faster than this |
| `std_ms` | ms | CUDA Event | Variation in GPU kernel times |
| `cv` | ratio | std_ms / mean_ms | Coefficient of variation (stability) |
| `eager_probe_ms` | ms | CUDA Event | Non-graph GPU time (for comparison) |
| `speedup_vs_baseline` | ratio | mean_ms / baseline_ms | How many times faster |

### 2.4 Important: duration_ms vs mean_ms

```
duration_ms (Python wall-clock)
│
├─ Create CUDA events
├─ Call generated_fn (actual GPU work measured)  ← mean_ms
├─ Record CUDA events
├─ Synchronize GPU
├─ Data transfer overhead
└─ Python function call overhead
```

**Usually**: `duration_ms` >> `mean_ms` (because duration includes host overhead)

---

## 3. Pipeline Integration

### 3.1 Pipeline Request Structure

```python
run_eval_pipeline({
    "sample_id": 1,
    "reference_code": "...",  # Reference kernel code
    "generated_code": "...",  # Generated kernel code

    # Compile configuration
    "compile": {
        "language": "python"
    },

    # Timing configuration (optional, skipped if not provided)
    "timing": {
        "generated_fn": compiled_fn,  # Callable function
        "warmup_iters": 3,
        "graph_iters": 10,
        "num_replays": 5
    },

    # Hack check configuration
    "hack_check": {},

    # Evaluation gates
    "gates": {
        "require_compile_ok": True,
        "require_timing_ok": False,  # Can skip timing
        "require_hack_clean": False   # Disabled (always pass)
    }
})
```

### 3.2 Pipeline Output Summary

```python
{
    "ok": True,
    "status": "passed",
    "sample_id": 1,
    "duration_ms": 150,  # Total pipeline time

    "summary": {
        "compile": {
            "return_code": 0,
            "error_category": "none",
            "warnings_count": 0,
            "errors_count": 0
        },
        "timing": {
            # All timing metrics extracted here
            "mean_ms": 12.345,
            "median_ms": 12.340,
            "p10_ms": 12.100,
            "p90_ms": 12.600,
            "generated_mean_ms": 12.345,
            "reference_mean_ms": 15.678,
            "speedup_vs_baseline": 1.269,
            "performance_verdict": "faster"
        },
        "hack_check": {
            "has_issue": False,
            "issue_total": 0,
            "torch_reference_count": 10,
            "torch_symbol_total": 12
        }
    }
}
```

---

## 4. Example Usage

### 4.1 Compilation Example

```python
from kernelrl.eval.pipeline import run_compile_module

result = run_compile_module({
    "language": "python",
    "code": """
def matrix_multiply(A, B):
    import torch
    return torch.matmul(A, B)
""",
    "timeout_sec": 30
})

print(f"Compilation OK: {result['ok']}")
print(f"Return code: {result['metrics']['return_code']}")
print(f"Duration: {result['duration_ms']}ms")
```

### 4.2 Timing Example

```python
from kernelrl.eval.pipeline import run_timing_module
import torch

def my_kernel(A, B):
    return torch.matmul(A, B)

def baseline_kernel(A, B):
    return torch.matmul(A, B)

result = run_timing_module({
    "generated_fn": my_kernel,
    "generated_fn_kwargs": {"A": torch.randn(2048, 2048), "B": torch.randn(2048, 2048)},
    "reference_fn": baseline_kernel,
    "reference_fn_kwargs": {"A": torch.randn(2048, 2048), "B": torch.randn(2048, 2048)},
    "warmup_iters": 5,
    "graph_iters": 20,
    "num_replays": 10
})

print(f"Generated mean: {result['metrics']['generated_mean_ms']:.3f}ms")
print(f"Reference mean: {result['metrics']['reference_mean_ms']:.3f}ms")
print(f"Speedup: {result['artifacts']['performance']['speedup_vs_baseline']:.2f}x")
```

---

## 5. Key Takeaways

### Compilation
- **Input**: Code string + language + compile config
- **Output**: Detailed metrics + build artifacts + error diagnostics
- **Local**: subprocess-based execution
- **Timeout**: Configurable (default 60s)

### Timing
- **Input**: Callable function + CUDA Graph parameters
- **Output**: GPU kernel time (mean, median, p10, p90, std, cv)
- **GPU-native**: torch.cuda.Event for precision
- **Comparison**: Optional baseline for speedup calculation
- **Key metric**: `mean_ms` is actual GPU time, `duration_ms` includes overhead

### Integration
- **Pipeline**: Chains compile → hack_check → timing
- **Summary**: Aggregates all metrics for high-level analysis
- **Flexible**: Can enable/disable timing, hack checks via gates
