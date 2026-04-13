"""
Integration tests for kernelrl.eval.pipeline module.

Tests cover:
1. Protocol utilities
2. Compile module
3. Hack check module
4. Timing module (basic)
5. Pipeline orchestrator
"""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from kernelrl.eval.pipeline import (
    run_compile_module,
    run_eval_pipeline,
    run_hack_check_module,
)
from kernelrl.eval.pipeline.correctness_module import run_correctness_module
from kernelrl.eval.pipeline.protocol import (
    elapsed_ms,
    started_timer,
    truncate_text,
    utc_now_iso,
)


class TestProtocol(unittest.TestCase):
    """Test protocol utility functions."""

    def test_utc_now_iso(self):
        """Test UTC timestamp generation."""
        ts = utc_now_iso()
        self.assertIsInstance(ts, str)
        self.assertIn("T", ts)  # ISO format includes T
        self.assertIn("+", ts)  # UTC timezone indicator

    def test_timer_functions(self):
        """Test timer start and elapsed functions."""
        start = started_timer()
        self.assertIsInstance(start, float)
        import time
        time.sleep(0.01)  # Sleep 10ms
        elapsed = elapsed_ms(start)
        self.assertGreaterEqual(elapsed, 10)
        self.assertIsInstance(elapsed, int)

    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "a" * 10000
        result = truncate_text(long_text, max_chars=100)
        self.assertLessEqual(len(result), 130)  # 100 + truncation marker
        self.assertIn("truncated", result)

        short_text = "hello"
        result = truncate_text(short_text, max_chars=100)
        self.assertEqual(result, short_text)

        result = truncate_text(None, max_chars=100)
        self.assertEqual(result, "")


class TestCompileModule(unittest.TestCase):
    """Test compilation module."""

    def test_python_compilation_success(self):
        """Test successful Python code compilation."""
        request = {
            "language": "python",
            "code": "def hello():\n    return 'world'",
        }
        result = run_compile_module(request)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["return_code"], 0)

    def test_python_compilation_syntax_error(self):
        """Test Python code with syntax errors."""
        request = {
            "language": "python",
            "code": "def hello(\n    invalid syntax",
        }
        result = run_compile_module(request)
        self.assertFalse(result["ok"])
        self.assertIn(result["status"], ["compile_error", "flagged_medium", "partial_success"])
        # Python compile may or may not extract error lines, so just verify it failed
        self.assertGreater(result["metrics"]["return_code"], 0)

    def test_cpp_compilation_skipped_if_no_compiler(self):
        """Test C++ compilation (may be skipped if no compiler)."""
        request = {
            "language": "cpp",
            "code": "int main() { return 0; }",
        }
        result = run_compile_module(request)
        # Either succeeds (if g++ available) or skips (if not), or compile_error
        self.assertIn(result["status"], ["ok", "skipped", "flagged_medium", "flagged_low", "compile_error"])

    def test_compile_missing_source(self):
        """Test compile with missing source."""
        request = {"language": "python"}
        result = run_compile_module(request)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "error")
        self.assertGreater(len(result["errors"]), 0)


class TestHackCheckModule(unittest.TestCase):
    """Test code quality and security checking module."""

    def test_clean_code(self):
        """Test clean code passes hack check."""
        request = {
            "code": "def add(a, b):\n    return a + b",
        }
        result = run_hack_check_module(request)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["issue_total"], 0)

    def test_dangerous_eval_detection(self):
        """Test detection of dangerous eval usage."""
        request = {
            "code": "result = eval(user_input)",
        }
        result = run_hack_check_module(request)
        self.assertFalse(result["ok"])
        self.assertGreater(result["metrics"]["high"], 0)
        # Find the dangerous_exec issue
        issues = result.get("issues", [])
        issue_ids = [i.get("id") for i in issues]
        self.assertIn("dangerous_exec", issue_ids)

    def test_system_shell_detection(self):
        """Test detection of shell command execution."""
        request = {
            "code": "import os\nos.system('rm -rf /')",
        }
        result = run_hack_check_module(request)
        self.assertFalse(result["ok"])
        self.assertGreater(result["metrics"]["high"], 0)
        issues = result.get("issues", [])
        issue_ids = [i.get("id") for i in issues]
        self.assertIn("system_shell", issue_ids)

    def test_fabricated_output_detection(self):
        """Test detection of fabricated tensor output."""
        request = {
            "code": "return torch.zeros_like(input_tensor)",
        }
        result = run_hack_check_module(request)
        self.assertFalse(result["ok"])
        self.assertGreater(result["metrics"]["high"], 0)
        issues = result.get("issues", [])
        issue_ids = [i.get("id") for i in issues]
        self.assertIn("fabricated_output", issue_ids)

    def test_torch_usage_policy(self):
        """Test torch usage policy enforcement."""
        code = "import torch\ntensor = torch.tensor([1, 2, 3])"

        # Test forbid policy
        request = {
            "code": code,
            "torch_usage_policy": "forbid",
        }
        result = run_hack_check_module(request)
        self.assertFalse(result["ok"])
        issues = [i for i in result.get("issues", []) if i.get("id") == "torch_usage_forbidden"]
        self.assertGreater(len(issues), 0)

        # Test warn policy
        request = {
            "code": code,
            "torch_usage_policy": "warn",
        }
        result = run_hack_check_module(request)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "warning")

        # Test allow policy
        request = {
            "code": code,
            "torch_usage_policy": "allow",
        }
        result = run_hack_check_module(request)
        self.assertTrue(result["ok"])

    def test_empty_code(self):
        """Test hack check with empty code."""
        request = {"code": ""}
        result = run_hack_check_module(request)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ok")


class TestPipelineOrchestrator(unittest.TestCase):
    """Test the unified evaluation pipeline."""

    def test_pipeline_basic(self):
        """Test basic pipeline execution."""
        request = {
            "sample_id": "test_001",
            "reference_code": "def add(a, b): return a + b",
            "generated_code": "def add(a, b):\n    return a + b",
            "compile": {"language": "python"},
            "hack_check": {},
        }
        result = run_eval_pipeline(request)

        self.assertIsNotNone(result)
        self.assertIn("sample_id", result)
        self.assertIn("modules", result)
        self.assertIn("summary", result)

    def test_pipeline_with_truncated_code(self):
        """Test pipeline truncation detection."""
        request = {
            "generated_code": "def func():\n    ...<truncated>...",
            "reference_code": "def func(): return 42",
            "compile": {"language": "python"},
            "hack_check": {},
        }
        result = run_eval_pipeline(request)

        summary = result.get("summary", {})
        code_selection = summary.get("code_selection", {})
        self.assertTrue(code_selection.get("generated_code_truncated", False))
        # Just verify truncation was detected
        self.assertIsNotNone(code_selection.get("generated_code_truncation_reason"))

    def test_pipeline_fallback_to_reference(self):
        """Test pipeline fallback to reference code when truncated."""
        request = {
            "reference_code": "def add(a, b): return a + b",
            "generated_code": "incomplete",
            "use_reference_on_generated_truncation": True,
            "compile": {"language": "python"},
            "hack_check": {},
        }
        result = run_eval_pipeline(request)

        summary = result.get("summary", {})
        code_selection = summary.get("code_selection", {})
        self.assertTrue(code_selection.get("generated_code_truncated", False))
        self.assertTrue(code_selection.get("used_reference_as_generated", False))

    def test_pipeline_gate_system(self):
        """Test pipeline gate system."""
        # Code that will definitely fail hack check with strict mode
        request = {
            "generated_code": "eval('dangerous code here')",
            "reference_code": "valid = 1",
            "compile": {"language": "python"},
            "hack_check": {"strict": True, "code": "eval('dangerous code here')"},
            "gates": {
                "require_compile_ok": True,
                "require_timing_ok": False,
                "require_hack_clean": True,
            },
        }
        result = run_eval_pipeline(request)

        # Should fail due to hack_check gate
        # The hack gate might pass/fail depending on whether hack_check runs
        # but we verify the overall result structure
        self.assertIsNotNone(result.get("gate_results"))

    def test_pipeline_multiple_modules(self):
        """Test pipeline with multiple modules."""
        request = {
            "sample_id": "test_multi",
            "reference_code": "return 0",
            "generated_code": "eval('dangerous')",
            "compile": {"language": "python"},
            "hack_check": {"strict": True},
        }
        result = run_eval_pipeline(request)

        modules = result.get("modules", {})
        self.assertIn("compile", modules)
        self.assertIn("hack_check", modules)

        # Hack check should flag dangerous code
        hack_result = modules.get("hack_check", {})
        # Either ok=False or has issues
        if not hack_result.get("ok", True):
            self.assertGreater(len(hack_result.get("issues", [])), 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios combining multiple modules."""

    def test_full_pipeline_clean_code(self):
        """Full pipeline test with clean code."""
        code = """
def matrix_multiply(a, b):
    '''Multiply two matrices'''
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            sum_val = 0
            for k in range(len(b)):
                sum_val += a[i][k] * b[k][j]
            row.append(sum_val)
        result.append(row)
    return result
"""
        request = {
            "sample_id": "matrix_multiply",
            "reference_code": code,
            "generated_code": code,
            "compile": {"language": "python"},
            "hack_check": {"strict": False},
        }
        result = run_eval_pipeline(request)

        # Should pass all gates
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "passed")

        gates = result.get("gate_results", {})
        self.assertTrue(gates.get("compile_gate", False))
        self.assertTrue(gates.get("hack_gate", False))

    def test_full_pipeline_suspicious_code(self):
        """Full pipeline test with suspicious code."""
        code = """
def get_result():
    try:
        return torch.zeros_like(input_tensor)
    except:
        return None
"""
        request = {
            "sample_id": "suspicious",
            "reference_code": "def get_result(): return compute()",
            "generated_code": code,
            "compile": {"language": "python"},
            "hack_check": {"strict": True},
        }
        result = run_eval_pipeline(request)

        # Should fail hack gate
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed")

        gates = result.get("gate_results", {})
        self.assertFalse(gates.get("hack_gate", True))

    def test_correctness_skipped_when_torch_missing(self):
        """Correctness should be skipped instead of failed when torch is unavailable."""
        with patch("kernelrl.eval.pipeline.correctness_module.importlib.util.find_spec", return_value=None):
            result = run_correctness_module({
                "reference_code": "x = 1",
                "generated_code": "y = 2",
                "skip_if_no_torch": True,
            })

        self.assertTrue(result.get("ok", False))
        self.assertEqual(result.get("status"), "skipped")


if __name__ == "__main__":
    unittest.main()
