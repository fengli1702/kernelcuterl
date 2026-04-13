import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, patch


def _load_rollout_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "rollout" / "single_turn_rollout.py"
    spec = importlib.util.spec_from_file_location("single_turn_rollout_module", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout = _load_rollout_module()


def _base_args(**overrides):
    args = argparse.Namespace(
        base_url="http://127.0.0.1:30000/v1",
        model="test-model",
        n=1,
        temperature=0.8,
        max_tokens=2048,
        seed=123,
        top_p=0.95,
        timeout=30.0,
        multi_turn=False,
        max_turns=5,
        repair_temperature=0.2,
        history_max_chars=180000,
        feedback_max_chars=3000,
        use_pipeline=False,
        pipeline_compile=True,
        pipeline_timing=False,
        pipeline_correctness=True,
        pipeline_hack_check=True,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestExtractionLogic(unittest.TestCase):
    def test_extract_prefers_tail_after_last_think(self):
        text = (
            "<think>draft one</think>\n"
            "```python\n"
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return x + 1\n"
            "```\n"
            "intermediate notes\n"
            "</think>\n"
            "```python\n"
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return x * 2\n"
            "```\n"
        )
        code, method = rollout.extract_kernel_with_method(text)
        self.assertIsNotNone(code)
        self.assertIn("return x * 2", code)
        self.assertNotIn("return x + 1", code)
        self.assertEqual(method, "fenced_pref")

    def test_extract_multiple_fenced_prefers_model_block(self):
        helper_block = "\n".join(["print('helper line')"] * 80)
        text = (
            "</think>\n"
            "```python\n"
            f"{helper_block}\n"
            "```\n"
            "```python\n"
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return x\n"
            "```\n"
        )
        code, method = rollout.extract_kernel_with_method(text)
        self.assertIsNotNone(code)
        self.assertIn("class ModelNew", code)
        self.assertEqual(method, "fenced_pref")


class TestFeedbackAndHistory(unittest.TestCase):
    def test_feedback_includes_full_previous_code_and_truncates_diagnostics(self):
        long_err = "E" * 1200
        long_issue = "I" * 900
        kernel_code = (
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return x\n\n"
            "def get_inputs():\n"
            "    return []\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        )
        eval_result = {
            "compile_ok": False,
            "correctness_ok": False,
            "hack_clean": False,
            "pipeline_result": {
                "summary": {
                    "compile": {"error_category": "syntax"},
                    "correctness": {"error_type": "mismatch", "max_diff": 1.0, "mean_diff": 0.5},
                },
                "modules": {
                    "compile": {
                        "errors": [{"message": long_err}],
                        "artifacts": {"stderr": ""},
                    },
                    "correctness": {"errors": [{"message": long_err}]},
                    "hack_check": {"issues": [{"id": "dangerous_exec", "message": long_issue}]},
                },
            },
        }
        feedback = rollout._build_feedback_message(
            prompt="Optimize this model",
            kernel_code=kernel_code,
            eval_result=eval_result,
            feedback_max_chars=220,
        )
        self.assertIn("Previous extracted code:", feedback)
        self.assertIn(kernel_code, feedback)
        self.assertIn("...(truncated)", feedback)
        self.assertIn("Current gate status:", feedback)

    def test_history_pruner_keeps_prompt_and_drops_oldest_turns(self):
        messages = [
            {"role": "user", "content": "PROMPT"},
            {"role": "assistant", "content": "OLD_ASSISTANT_" * 220},
            {"role": "user", "content": "OLD_FEEDBACK_" * 220},
            {"role": "assistant", "content": "NEW_ASSISTANT_" * 120},
            {"role": "user", "content": "NEW_FEEDBACK_" * 120},
        ]
        max_chars = 5000
        pruned = rollout.prune_message_history(messages, max_chars=max_chars)
        merged = "\n".join(m["content"] for m in pruned)
        self.assertEqual(pruned[0]["content"], "PROMPT")
        self.assertNotIn("OLD_ASSISTANT_", merged)
        self.assertNotIn("OLD_FEEDBACK_", merged)
        self.assertIn("NEW_ASSISTANT_", merged)
        self.assertIn("NEW_FEEDBACK_", merged)
        self.assertLessEqual(sum(len(m["content"]) for m in pruned), max_chars)


class TestMultiTurnFlow(unittest.IsolatedAsyncioTestCase):
    async def test_multi_turn_stops_only_when_compile_correctness_hack_all_pass(self):
        args = _base_args(
            multi_turn=True,
            n=1,
            max_turns=5,
            use_pipeline=True,
        )

        with patch.object(
            rollout,
            "bounded_chat_completion",
            new=AsyncMock(side_effect=[["turn1"], ["turn2"]]),
        ), patch.object(
            rollout,
            "extract_kernel_with_method",
            side_effect=lambda text: (f"code_{text}", "mock_extract"),
        ), patch.object(
            rollout,
            "evaluate_kernel_with_pipeline",
            side_effect=lambda code, *_args, **_kwargs: (
                {
                    "correct": True,
                    "compile_ok": True,
                    "correctness_ok": True,
                    "hack_clean": False,
                    "pipeline_result": {},
                }
                if code == "code_turn1"
                else {
                    "correct": True,
                    "compile_ok": True,
                    "correctness_ok": True,
                    "hack_clean": True,
                    "pipeline_result": {},
                }
            ),
        ):
            result = await rollout.process_item(
                idx=0,
                prompt="task",
                reference="ref",
                sem=asyncio.Semaphore(1),
                client=object(),
                args=args,
            )

        sample = result.samples[0]
        self.assertEqual(len(sample.turns or []), 2)
        self.assertEqual(sample.turns[-1].get("stop_reason"), "success_gate")
        self.assertIsNotNone(sample.turns[0].get("feedback_to_model"))
        self.assertIsNone(sample.error)
        self.assertTrue(sample.eval_result.get("compile_ok"))
        self.assertTrue(sample.eval_result.get("correctness_ok"))
        self.assertTrue(sample.eval_result.get("hack_clean"))

    async def test_multi_turn_n2_trajectories_iterate_independently(self):
        args = _base_args(
            multi_turn=True,
            n=2,
            max_turns=4,
            use_pipeline=True,
        )
        chat_side_effect = [
            ["a1", "b1"],  # turn1, n=2
            ["a2"],        # trajectory A repair
            ["b2"],        # trajectory B repair turn2
            ["b3"],        # trajectory B repair turn3
        ]

        def fake_eval(code, *_args, **_kwargs):
            table = {
                "code_a1": {"correct": False, "compile_ok": False, "correctness_ok": False, "hack_clean": True},
                "code_a2": {"correct": True, "compile_ok": True, "correctness_ok": True, "hack_clean": True},
                "code_b1": {"correct": False, "compile_ok": True, "correctness_ok": False, "hack_clean": True},
                "code_b2": {"correct": False, "compile_ok": True, "correctness_ok": True, "hack_clean": False},
                "code_b3": {"correct": True, "compile_ok": True, "correctness_ok": True, "hack_clean": True},
            }
            item = dict(table[code])
            item["pipeline_result"] = {}
            return item

        with patch.object(
            rollout,
            "bounded_chat_completion",
            new=AsyncMock(side_effect=chat_side_effect),
        ), patch.object(
            rollout,
            "extract_kernel_with_method",
            side_effect=lambda text: (f"code_{text}", "mock_extract"),
        ), patch.object(
            rollout,
            "evaluate_kernel_with_pipeline",
            side_effect=fake_eval,
        ):
            result = await rollout.process_item(
                idx=1,
                prompt="task",
                reference="ref",
                sem=asyncio.Semaphore(1),
                client=object(),
                args=args,
            )

        self.assertEqual(len(result.samples), 2)
        sample_a, sample_b = result.samples
        self.assertEqual(len(sample_a.turns or []), 2)
        self.assertEqual(len(sample_b.turns or []), 3)
        self.assertEqual(sample_a.turns[-1].get("stop_reason"), "success_gate")
        self.assertEqual(sample_b.turns[-1].get("stop_reason"), "success_gate")
        self.assertEqual(sample_a.text, "a2")
        self.assertEqual(sample_b.text, "b3")

    async def test_single_turn_regression_output_shape(self):
        args = _base_args(
            multi_turn=False,
            n=1,
            use_pipeline=False,
        )
        with patch.object(
            rollout,
            "bounded_chat_completion",
            new=AsyncMock(return_value=["single"]),
        ), patch.object(
            rollout,
            "extract_kernel",
            return_value="code_single",
        ), patch.object(
            rollout,
            "evaluate_kernel",
            return_value={"correct": True},
        ):
            result = await rollout.process_item(
                idx=2,
                prompt="task",
                reference="ref",
                sem=asyncio.Semaphore(1),
                client=object(),
                args=args,
            )

        self.assertEqual(len(result.samples), 1)
        sample = result.samples[0]
        self.assertEqual(sample.turns, [])
        self.assertTrue(sample.eval_result.get("correct"))
        self.assertNotIn("mean_turns_used", result.summary)
        self.assertEqual(result.summary.get("num_samples"), 1)
        self.assertEqual(result.summary.get("num_correct"), 1)


if __name__ == "__main__":
    unittest.main()
