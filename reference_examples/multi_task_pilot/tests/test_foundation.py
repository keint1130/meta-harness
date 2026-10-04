from __future__ import annotations

import base64
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

from multi_task_pilot.baseline import BaselineHarness, load_harness_spec
from multi_task_pilot.coding_evaluation import evaluate_livecodebench
from multi_task_pilot.docker_sandbox import DockerSandbox
from multi_task_pilot.contracts import EvalCase, Task, TaskFamily
from multi_task_pilot.evaluation import (
    Score,
    macro_average,
    score_math,
    score_popqa,
)
from multi_task_pilot.splits import fixed_split
from multi_task_pilot.tools import FrozenTools, ToolBudgetExceeded, ToolBudgets


class SplitTests(unittest.TestCase):
    def test_fixed_split_is_stable_and_disjoint(self) -> None:
        rows = [{"id": str(i)} for i in range(20)]
        first = fixed_split(rows, id_field="id", seed=42, dev_size=5, test_size=7)
        second = fixed_split(
            reversed(rows), id_field="id", seed=42, dev_size=5, test_size=7
        )
        self.assertEqual(first, second)
        self.assertFalse(
            {row["id"] for row in first["dev"]}
            & {row["id"] for row in first["test"]}
        )


class ContractTests(unittest.TestCase):
    def test_harness_task_does_not_contain_reference_answer(self) -> None:
        task = Task(id="q1", family=TaskFamily.MATH, prompt="2+2")
        case = EvalCase(task=task, reference="4")
        self.assertFalse(hasattr(case.task, "reference"))

    def test_baseline_passes_reasoning_effort_without_references(self) -> None:
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message={"role": "assistant", "content": "4"})],
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            )

        harness = BaselineHarness(
            model="openai/gpt-6-luna",
            reasoning_effort="medium",
            completion_fn=fake_completion,
        )
        result = harness.solve(
            Task(id="q", family=TaskFamily.MATH, prompt="2+2"),
            FrozenTools(resources={}),
        )
        self.assertEqual(result.answer, "4")
        self.assertEqual(captured["reasoning_effort"], "medium")
        self.assertNotIn("reference", json.dumps(captured["messages"]))

    def test_committed_baseline_harness_spec_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = load_harness_spec(root / "harness_specs" / "baseline.json")
        self.assertEqual(spec["name"], "baseline")


class EvaluationTests(unittest.TestCase):
    def test_popqa_accepts_alias_in_short_answer(self) -> None:
        score = score_popqa("The answer is political leader.", ["politician", "political leader"])
        self.assertTrue(score.correct)

    def test_math_extracts_boxed_answer(self) -> None:
        self.assertTrue(score_math(r"Therefore, \boxed{42}", "42").correct)

    def test_math_accepts_symbolically_equivalent_fraction(self) -> None:
        self.assertTrue(score_math(r"\boxed{\frac{1}{2}}", "0.5").correct)

    def test_math_does_not_reduce_radical_reference_to_leading_number(self) -> None:
        self.assertFalse(
            score_math(r"\boxed{2\sqrt{106}\text{ feet}}", r"2\sqrt{113}").correct
        )

    def test_math_accepts_equivalent_complex_number(self) -> None:
        self.assertTrue(score_math(r"\boxed{-2+7i}", "-2 + 7i").correct)

    def test_math_accepts_formatted_large_integer(self) -> None:
        self.assertTrue(
            score_math(r"\boxed{11111111100}", r"11,\! 111,\! 111,\! 100").correct
        )

    def test_math_accepts_text_category(self) -> None:
        self.assertTrue(score_math(r"\boxed{\text{even}}", r"\text{even}").correct)

    def test_macro_average_weights_families_equally(self) -> None:
        scores = {
            TaskFamily.CODING: [Score(True, 1.0), Score(False, 0.0)],
            TaskFamily.RETRIEVAL: [Score(True, 1.0)],
            TaskFamily.MATH: [Score(False, 0.0)],
        }
        self.assertAlmostEqual(macro_average(scores), 0.5)


class DockerCommandTests(unittest.TestCase):
    def test_command_disables_network_and_drops_privileges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = DockerSandbox(docker_executable="docker").command(Path(tmp))
        self.assertIn("none", command)
        self.assertIn("ALL", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--read-only", command)
        self.assertIn("--interactive", command)


class ToolTests(unittest.TestCase):
    def test_retrieve_uses_only_frozen_documents_and_enforces_budget(self) -> None:
        tools = FrozenTools(
            resources={"documents": ["Paris is in France.", "Tokyo is in Japan."]},
            budgets=ToolBudgets(retrieve_calls=1, retrieved_documents=1),
        )
        self.assertIn("Tokyo", tools.retrieve("Japan capital", top_k=5)[0]["text"])
        with self.assertRaises(ToolBudgetExceeded):
            tools.retrieve("France")

    def test_python_exec_runs_in_docker(self) -> None:
        tools = FrozenTools(budgets=ToolBudgets(python_calls=1))
        self.assertEqual(tools.python_exec("print(6 * 7)").strip(), "42")
        with self.assertRaises(ToolBudgetExceeded):
            tools.python_exec("print('again')")


class CodingEvaluationTests(unittest.TestCase):
    def test_functional_problem_passes_public_and_private_tests(self) -> None:
        private = [{"input": "40\n2", "output": "42", "testtype": "functional"}]
        encoded = base64.b64encode(
            zlib.compress(pickle.dumps(json.dumps(private)))
        ).decode()
        case = EvalCase(
            task=Task(
                id="sum",
                family=TaskFamily.CODING,
                prompt="Add two numbers",
                metadata={
                    "public_test_cases": [
                        {"input": "2\n3", "output": "5", "testtype": "functional"}
                    ]
                },
            ),
            reference={
                "private_test_cases_encoded": encoded,
                "metadata": {"func_name": "add"},
            },
        )
        answer = """```python
class Solution:
    def add(self, a, b):
        return a + b
```"""
        result = evaluate_livecodebench(case, answer, DockerSandbox())
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.tests, 2)


if __name__ == "__main__":
    unittest.main()
