"""
Correctness checking module for evaluating generated code against reference.

Executes both reference and generated code with same inputs and compares outputs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Any

from .protocol import (
    add_error,
    add_issue,
    finalize_module_result,
    init_module_result,
    started_timer,
    truncate_text,
)


def _create_test_script(ref_path: str, gen_path: str) -> str:
    """Create a Python script that tests both reference and generated code."""
    return f'''import hashlib
import os
import sys
import tempfile
import traceback
import torch

def _ensure_cxx17(flags):
    if flags is None:
        return ["-std=c++17"]
    if isinstance(flags, str):
        flags = [flags]

    normalized = []
    saw_std = False
    for flag in flags:
        if not isinstance(flag, str):
            normalized.append(flag)
            continue
        if flag.startswith("-std="):
            saw_std = True
            if flag in ("-std=c++11", "-std=gnu++11", "-std=c++14", "-std=gnu++14"):
                normalized.append("-std=c++17")
            else:
                normalized.append(flag)
        else:
            normalized.append(flag)

    if not saw_std:
        normalized.append("-std=c++17")
    return normalized

def _patch_load_inline():
    try:
        import torch.utils.cpp_extension as cpp_ext
    except Exception:
        return

    original = cpp_ext.load_inline

    def _wrapped_load_inline(*args, **kwargs):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        if isinstance(name, str) and name:
            source_fingerprint = repr((
                kwargs.get("cpp_sources"),
                kwargs.get("cuda_sources"),
                kwargs.get("functions"),
            ))
            unique_suffix = hashlib.md5(source_fingerprint.encode("utf-8")).hexdigest()[:12]
            patched_name = f"{{name}}_{{unique_suffix}}"
            kwargs["name"] = patched_name
            if args:
                args = list(args)
                args[0] = patched_name
                args = tuple(args)

        kwargs["extra_cflags"] = _ensure_cxx17(kwargs.get("extra_cflags"))
        kwargs["extra_cuda_cflags"] = _ensure_cxx17(kwargs.get("extra_cuda_cflags"))
        return original(*args, **kwargs)

    cpp_ext.load_inline = _wrapped_load_inline

def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {{k: _to_device(v, device) for k, v in value.items()}}
    return value

def run_test():
    """Run both models and compare outputs."""
    try:
        # Isolate extension cache per test run to avoid stale/ABI collisions.
        os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="kernelrl_ext_")
        _patch_load_inline()

        # Execute reference code
        ref_globals = {{"torch": torch}}
        with open(r"{ref_path}", "r") as f:
            reference_code = f.read()
        exec(reference_code, ref_globals)

        # Execute generated code
        gen_globals = {{"torch": torch}}
        with open(r"{gen_path}", "r") as f:
            generated_code = f.read()
        exec(generated_code, gen_globals)

        # Get Model and ModelNew classes
        Model = ref_globals.get("Model")
        ModelNew = gen_globals.get("ModelNew")

        if Model is None:
            return {{"error": "reference_model_not_found", "message": "Model class not found in reference code"}}
        if ModelNew is None:
            return {{"error": "generated_model_not_found", "message": "ModelNew class not found in generated code"}}

        # Get input generation functions (prefer from reference)
        get_inputs = ref_globals.get("get_inputs") or gen_globals.get("get_inputs")
        get_init_inputs = ref_globals.get("get_init_inputs") or gen_globals.get("get_init_inputs")

        if get_inputs is None:
            return {{"error": "get_inputs_not_found", "message": "get_inputs() function not found"}}
        if get_init_inputs is None:
            return {{"error": "get_init_inputs_not_found", "message": "get_init_inputs() function not found"}}

        # Initialize models
        init_inputs = get_init_inputs()
        if not isinstance(init_inputs, (list, tuple)):
            init_inputs = [init_inputs] if init_inputs is not None else []

        model_ref = Model(*init_inputs)
        model_gen = ModelNew(*init_inputs)

        # Keep both models and inputs on the same device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ref = model_ref.to(device)
        model_gen = model_gen.to(device)

        # Get test inputs
        inputs = get_inputs()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        inputs = _to_device(inputs, device)

        # Run reference model
        model_ref.eval()
        with torch.no_grad():
            output_ref = model_ref(*inputs)

        # Run generated model
        model_gen.eval()
        with torch.no_grad():
            output_gen = model_gen(*inputs)

        # Compare outputs
        if isinstance(output_ref, torch.Tensor) and isinstance(output_gen, torch.Tensor):
            # Single tensor output
            if output_ref.shape != output_gen.shape:
                return {{
                    "correct": False,
                    "error": "shape_mismatch",
                    "message": f"Output shape mismatch: ref={{output_ref.shape}}, gen={{output_gen.shape}}"
                }}

            # Use allclose for numerical comparison
            rtol = 1e-2
            atol = 1e-2
            matches = torch.allclose(output_ref, output_gen, rtol=rtol, atol=atol)

            if matches:
                return {{
                    "correct": True,
                    "message": "Outputs match within tolerance",
                    "rtol": rtol,
                    "atol": atol,
                    "max_diff": float(torch.max(torch.abs(output_ref - output_gen)).item()),
                    "mean_diff": float(torch.mean(torch.abs(output_ref - output_gen)).item()),
                }}
            else:
                max_diff = float(torch.max(torch.abs(output_ref - output_gen)).item())
                mean_diff = float(torch.mean(torch.abs(output_ref - output_gen)).item())
                return {{
                    "correct": False,
                    "error": "output_mismatch",
                    "message": f"Outputs do not match: max_diff={{max_diff:.6e}}, mean_diff={{mean_diff:.6e}}",
                    "rtol": rtol,
                    "atol": atol,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                }}

        elif isinstance(output_ref, (list, tuple)) and isinstance(output_gen, (list, tuple)):
            # Multiple tensor outputs
            if len(output_ref) != len(output_gen):
                return {{
                    "correct": False,
                    "error": "output_count_mismatch",
                    "message": f"Output count mismatch: ref={{len(output_ref)}}, gen={{len(output_gen)}}"
                }}

            rtol = 1e-2
            atol = 1e-2
            all_match = True
            max_diff_overall = 0.0
            mean_diff_overall = 0.0

            for i, (ref_out, gen_out) in enumerate(zip(output_ref, output_gen)):
                if not isinstance(ref_out, torch.Tensor) or not isinstance(gen_out, torch.Tensor):
                    continue

                if ref_out.shape != gen_out.shape:
                    return {{
                        "correct": False,
                        "error": "shape_mismatch",
                        "message": f"Output[{{i}}] shape mismatch: ref={{ref_out.shape}}, gen={{gen_out.shape}}"
                    }}

                matches = torch.allclose(ref_out, gen_out, rtol=rtol, atol=atol)
                if not matches:
                    all_match = False

                max_diff = float(torch.max(torch.abs(ref_out - gen_out)).item())
                mean_diff = float(torch.mean(torch.abs(ref_out - gen_out)).item())
                max_diff_overall = max(max_diff_overall, max_diff)
                mean_diff_overall += mean_diff / len(output_ref)

            if all_match:
                return {{
                    "correct": True,
                    "message": "All outputs match within tolerance",
                    "rtol": rtol,
                    "atol": atol,
                    "num_outputs": len(output_ref),
                    "max_diff": max_diff_overall,
                    "mean_diff": mean_diff_overall,
                }}
            else:
                return {{
                    "correct": False,
                    "error": "output_mismatch",
                    "message": f"Some outputs do not match: max_diff={{max_diff_overall:.6e}}, mean_diff={{mean_diff_overall:.6e}}",
                    "rtol": rtol,
                    "atol": atol,
                    "num_outputs": len(output_ref),
                    "max_diff": max_diff_overall,
                    "mean_diff": mean_diff_overall,
                }}

        else:
            return {{
                "error": "unsupported_output_type",
                "message": f"Unsupported output types: ref={{type(output_ref).__name__}}, gen={{type(output_gen).__name__}}"
            }}

    except Exception as exc:
        message = str(exc)
        if len(message) > 2000:
            message = message[:2000] + "...(truncated)"
        return {{
            "error": "execution_failed",
            "message": message,
            "traceback": traceback.format_exc()
        }}

if __name__ == "__main__":
    result = run_test()

    # Print result as Python repr for easy parsing
    print("__RESULT_START__")
    print(repr(result))
    print("__RESULT_END__")
'''


def run_correctness_module(request: dict[str, Any]) -> dict[str, Any]:
    """
    Run correctness checking module.

    Request schema:
    - reference_code: str - Reference implementation
    - generated_code: str - Generated implementation to test
    - timeout_sec: int - Execution timeout (default: 120)

    Returns:
        Module result dict with:
        - ok: bool - Whether generated code is correct
        - status: str - "ok", "failed", "error", "skipped"
        - metrics: dict - Correctness metrics
        - artifacts: dict - Detailed comparison results
    """
    result = init_module_result("correctness", request)
    start = started_timer()

    reference_code = request.get("reference_code")
    generated_code = request.get("generated_code")

    if not reference_code or not isinstance(reference_code, str):
        add_error(result, "reference_code_missing", "Reference code is required")
        return finalize_module_result(result, start, ok=False, status="error")

    if not generated_code or not isinstance(generated_code, str):
        add_error(result, "generated_code_missing", "Generated code is required")
        return finalize_module_result(result, start, ok=False, status="error")

    # Environment dependency guard:
    # Most correctness checks require torch; if unavailable, skip gracefully to
    # avoid environment-related failures in environments without torch.
    if request.get("skip_if_no_torch", True):
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            result["metrics"]["correct"] = False
            result["metrics"]["error_type"] = "environment_dependency_missing"
            result["metrics"]["environment_dependency_missing"] = "torch"
            result["artifacts"]["skip_reason"] = "torch is not installed in current environment"
            add_issue(
                result,
                "correctness_skipped_due_to_environment",
                "low",
                "Torch is required by this correctness check but is not installed",
            )
            return finalize_module_result(result, start, ok=True, status="skipped")

    timeout_sec = int(request.get("timeout_sec", 120))

    # Create temporary files for reference and generated code
    ref_file = None
    gen_file = None
    script_file = None

    try:
        # Write reference code to temp file
        ref_file = tempfile.NamedTemporaryFile(mode='w', suffix='_ref.py', delete=False)
        ref_file.write(reference_code)
        ref_file.close()

        # Write generated code to temp file
        gen_file = tempfile.NamedTemporaryFile(mode='w', suffix='_gen.py', delete=False)
        gen_file.write(generated_code)
        gen_file.close()

        # Create test script
        test_script = _create_test_script(ref_file.name, gen_file.name)

        # Write test script to temporary file
        script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        script_file.write(test_script)
        script_file.close()
        script_path = Path(script_file.name)

        try:
            # Run test script
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )

            # Parse output
            stdout = proc.stdout
            stderr = proc.stderr

            result["artifacts"]["stdout"] = truncate_text(stdout, max_chars=2000)
            result["artifacts"]["stderr"] = truncate_text(stderr, max_chars=2000)
            result["artifacts"]["return_code"] = proc.returncode

            if proc.returncode != 0:
                add_error(
                    result,
                    "test_script_failed",
                    f"Test script exited with code {proc.returncode}",
                    details={"stderr": truncate_text(stderr, max_chars=500)}
                )
                return finalize_module_result(result, start, ok=False, status="error")

            # Extract result from output
            if "__RESULT_START__" in stdout and "__RESULT_END__" in stdout:
                result_str = stdout.split("__RESULT_START__")[1].split("__RESULT_END__")[0].strip()
                try:
                    # Use eval to parse the repr output
                    test_result = eval(result_str)
                except Exception as exc:
                    add_error(result, "result_parse_failed", f"Failed to parse test result: {exc}")
                    return finalize_module_result(result, start, ok=False, status="error")
            else:
                add_error(result, "result_not_found", "Test result markers not found in output")
                return finalize_module_result(result, start, ok=False, status="error")

            # Process test result
            correct = test_result.get("correct", False)
            error = test_result.get("error")
            message = test_result.get("message", "")

            result["metrics"]["correct"] = correct
            result["artifacts"]["test_result"] = test_result

            if error:
                add_error(result, error, message)
                result["metrics"]["error_type"] = error
                return finalize_module_result(result, start, ok=False, status="failed")

            if correct:
                # Add metrics
                result["metrics"]["rtol"] = test_result.get("rtol")
                result["metrics"]["atol"] = test_result.get("atol")
                result["metrics"]["max_diff"] = test_result.get("max_diff")
                result["metrics"]["mean_diff"] = test_result.get("mean_diff")
                result["metrics"]["num_outputs"] = test_result.get("num_outputs", 1)

                return finalize_module_result(result, start, ok=True, status="ok")
            else:
                # Incorrect output
                result["metrics"]["rtol"] = test_result.get("rtol")
                result["metrics"]["atol"] = test_result.get("atol")
                result["metrics"]["max_diff"] = test_result.get("max_diff")
                result["metrics"]["mean_diff"] = test_result.get("mean_diff")
                result["metrics"]["num_outputs"] = test_result.get("num_outputs", 1)

                add_issue(
                    result,
                    "output_mismatch",
                    "high",
                    message,
                    details={
                        "max_diff": test_result.get("max_diff"),
                        "mean_diff": test_result.get("mean_diff"),
                    }
                )
                return finalize_module_result(result, start, ok=False, status="failed")

        finally:
            # Clean up temporary files
            for tmp_file in [script_path if 'script_path' in locals() else None,
                           Path(ref_file.name) if ref_file else None,
                           Path(gen_file.name) if gen_file else None]:
                if tmp_file:
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass

    except subprocess.TimeoutExpired:
        add_error(result, "execution_timeout", f"Test execution timed out after {timeout_sec}s")
        # Clean up temp files
        for tmp_file in [Path(ref_file.name) if ref_file else None,
                       Path(gen_file.name) if gen_file else None]:
            if tmp_file:
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
        return finalize_module_result(result, start, ok=False, status="error")

    except Exception as exc:
        add_error(result, "unexpected_exception", str(exc))
        # Clean up temp files
        for tmp_file in [Path(ref_file.name) if ref_file else None,
                       Path(gen_file.name) if gen_file else None]:
            if tmp_file:
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
        return finalize_module_result(result, start, ok=False, status="error")
