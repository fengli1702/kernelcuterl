"""
Compilation module for evaluating kernel code.

Supports Python, C++, and CUDA compilation with error classification and diagnostics.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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


_EXTENSION_MAP = {
    "python": ".py",
    "py": ".py",
    "cuda": ".cu",
    "cpp": ".cpp",
    "c++": ".cpp",
    "c": ".c",
}

_WARNING_RE = re.compile(r"\bwarning\b", flags=re.IGNORECASE)
_ERROR_RE = re.compile(r"\berror\b", flags=re.IGNORECASE)


def _default_filename(language: str) -> str:
    ext = _EXTENSION_MAP.get(language.lower(), ".txt")
    return f"candidate{ext}"


def _resolve_compile_cmd(language: str, source_path: Path, build_dir: Path) -> tuple[str | None, str | None]:
    lang = language.lower()

    if lang in {"python", "py"}:
        cmd = f"{shlex.quote(sys.executable)} -m py_compile {shlex.quote(str(source_path))}"
        return cmd, None

    if lang == "cuda":
        nvcc = shutil.which("nvcc")
        if not nvcc:
            return None, "nvcc_not_found"
        out_obj = build_dir / "candidate.o"
        cmd = f"{shlex.quote(nvcc)} -c {shlex.quote(str(source_path))} -o {shlex.quote(str(out_obj))}"
        return cmd, None

    if lang in {"cpp", "c++", "c"}:
        gxx = shutil.which("g++")
        if not gxx:
            return None, "gxx_not_found"
        out_obj = build_dir / "candidate.o"
        cmd = f"{shlex.quote(gxx)} -c {shlex.quote(str(source_path))} -o {shlex.quote(str(out_obj))}"
        return cmd, None

    return None, "no_default_compile_cmd"


def _classify_compile_error(stderr: str, return_code: int) -> str:
    s = (stderr or "").lower()
    if return_code == 127:
        return "tool_not_found_or_not_executable"
    if "no such file or directory" in s:
        return "missing_include_or_source"
    if "undefined reference" in s:
        return "link_error"
    if "syntax error" in s or "expected" in s:
        return "syntax_error"
    if "permission denied" in s:
        return "permission_error"
    if "out of memory" in s:
        return "oom"
    return "compile_error"


def _extract_diag_lines(stderr: str, max_items: int = 50) -> dict[str, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lines = (stderr or "").splitlines()

    for i, line in enumerate(lines, start=1):
        line_strip = line.strip()
        if not line_strip:
            continue
        if _WARNING_RE.search(line_strip):
            warnings.append({"line_no": i, "text": line_strip})
        if _ERROR_RE.search(line_strip):
            errors.append({"line_no": i, "text": line_strip})
        if len(warnings) + len(errors) >= max_items:
            break

    return {"warnings": warnings, "errors": errors}


def _collect_build_outputs(build_dir: Path, limit: int = 40) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    if not build_dir.exists():
        return outputs

    for p in sorted(build_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(build_dir))
        outputs.append({"path": rel, "size_bytes": p.stat().st_size})
        if len(outputs) >= limit:
            break
    return outputs


def run_compile_module(request: dict[str, Any]) -> dict[str, Any]:
    """
    Compile module interface (dict in, dict out).

    Request keys:
    - language: str
    - code: str (optional)
    - source_path: str (optional)
    - build_dir: str (optional)
    - filename: str (optional)
    - compile_cmd: str (optional)
    - timeout_sec: int (optional, default 60)
    - env: dict[str, str] (optional)
    - shell: bool (optional, default True)
    - capture_output_chars: int (optional, default 4000)
    - collect_build_outputs: bool (optional, default True)
    """
    start = started_timer()
    result = init_module_result("compile", request)

    try:
        language = str(request.get("language", "python")).lower()
        timeout_sec = int(request.get("timeout_sec", 60))
        shell = bool(request.get("shell", True))
        capture_output_chars = int(request.get("capture_output_chars", 4000))
        collect_build_outputs = bool(request.get("collect_build_outputs", True))

        build_dir = Path(request.get("build_dir") or tempfile.mkdtemp(prefix="kernel_eval_build_"))
        build_dir.mkdir(parents=True, exist_ok=True)

        code = request.get("code")
        source_path_value = request.get("source_path")

        if source_path_value:
            source_path = Path(source_path_value)
            if not source_path.exists():
                add_error(result, "source_not_found", f"source_path does not exist: {source_path}")
                return finalize_module_result(result, start, ok=False, status="error")
        elif isinstance(code, str) and code.strip():
            filename = str(request.get("filename") or _default_filename(language))
            source_path = build_dir / filename
            source_path.write_text(code, encoding="utf-8")
        else:
            add_error(result, "missing_source", "Either source_path or non-empty code must be provided")
            return finalize_module_result(result, start, ok=False, status="error")

        compile_cmd = request.get("compile_cmd")
        reason = None
        if not compile_cmd:
            compile_cmd, reason = _resolve_compile_cmd(language, source_path, build_dir)

        if not compile_cmd:
            add_issue(
                result,
                "compile_skipped",
                "medium",
                "Compile command is not available",
                details={"reason": reason, "language": language},
            )
            result["artifacts"].update(
                {
                    "source_path": str(source_path),
                    "build_dir": str(build_dir),
                    "diagnostics": {"warnings": [], "errors": []},
                }
            )
            return finalize_module_result(result, start, ok=False, status="skipped")

        env = os.environ.copy()
        user_env = request.get("env") or {}
        if isinstance(user_env, dict):
            env.update({str(k): str(v) for k, v in user_env.items()})

        proc = subprocess.run(
            compile_cmd,
            cwd=str(build_dir),
            shell=shell,
            timeout=timeout_sec,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        diags = _extract_diag_lines(stderr)

        warnings_count = len(diags["warnings"])
        errors_count = len(diags["errors"])
        error_category = _classify_compile_error(stderr, proc.returncode) if proc.returncode != 0 else "none"

        build_outputs = _collect_build_outputs(build_dir) if collect_build_outputs else []

        result["artifacts"].update(
            {
                "source_path": str(source_path),
                "build_dir": str(build_dir),
                "compile_cmd": compile_cmd,
                "stdout": truncate_text(stdout, max_chars=capture_output_chars),
                "stderr": truncate_text(stderr, max_chars=capture_output_chars),
                "diagnostics": diags,
                "build_outputs": build_outputs,
            }
        )
        result["metrics"].update(
            {
                "return_code": proc.returncode,
                "warnings_count": warnings_count,
                "errors_count": errors_count,
                "timeout_sec": timeout_sec,
                "error_category": error_category,
                "build_output_count": len(build_outputs),
                "compiled_artifact_present": any(x["path"].endswith((".o", ".so", ".pyd", ".dll")) for x in build_outputs),
            }
        )

        if proc.returncode == 0:
            if warnings_count > 0:
                add_issue(
                    result,
                    "compile_warning",
                    "low",
                    "Compilation succeeded with warnings",
                    details={"warnings_count": warnings_count},
                )
            return finalize_module_result(result, start, ok=True, status="ok")

        first_error = diags["errors"][0]["text"] if diags["errors"] else ""
        add_error(
            result,
            "compile_failed",
            "Compilation command returned non-zero exit code",
            details={
                "return_code": proc.returncode,
                "error_category": error_category,
                "first_error": first_error,
            },
        )
        return finalize_module_result(result, start, ok=False, status="compile_error")

    except subprocess.TimeoutExpired as exc:
        add_error(
            result,
            "compile_timeout",
            f"Compilation timed out after {exc.timeout}s",
        )
        return finalize_module_result(result, start, ok=False, status="timeout")
    except Exception as exc:  # pragma: no cover - defensive
        add_error(result, "unexpected_exception", str(exc))
        return finalize_module_result(result, start, ok=False, status="error")
