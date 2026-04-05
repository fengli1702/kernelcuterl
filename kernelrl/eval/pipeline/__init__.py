"""
KernelRL Evaluation Pipeline

Integrated evaluation pipeline for kernel code with:
- Timing/Benchmarking (CUDA graphs and command execution)
- Compilation (Python, C++, CUDA)
- Hack Checking (Code quality and security)
"""

from .protocol import (
    utc_now_iso,
    started_timer,
    elapsed_ms,
    truncate_text,
    init_module_result,
    finalize_module_result,
    add_issue,
    add_error,
)
from .timing_module import run_timing_module
from .compile_module import run_compile_module
from .hack_check_module import run_hack_check_module
from .pipeline import run_eval_pipeline

__all__ = [
    # Protocol utilities
    "utc_now_iso",
    "started_timer",
    "elapsed_ms",
    "truncate_text",
    "init_module_result",
    "finalize_module_result",
    "add_issue",
    "add_error",
    # Modules
    "run_timing_module",
    "run_compile_module",
    "run_hack_check_module",
    "run_eval_pipeline",
]
