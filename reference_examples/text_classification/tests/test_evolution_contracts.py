from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import meta_harness
from benchmark import run_dir as benchmark_run_dir
from text_classification.agents.fewshot_memory import FewShotMemory, _seed_for_input
from text_classification.data.api import _with_context


def _args(**overrides):
    values = {
        "model": "openai/gpt-6-luna",
        "run_name": "test-run",
        "fresh": False,
        "skip_baseline": True,
        "iterations": 0,
        "propose_timeout": 1,
        "test": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DataContractTests(unittest.TestCase):
    def test_with_context_preserves_metadata(self) -> None:
        examples = _with_context(
            [
                {
                    "input": "question",
                    "target": "answer",
                    "context": "instructions",
                    "raw_question": "question",
                    "source_id": "example-1",
                }
            ]
        )

        self.assertEqual(examples[0]["input"], "instructions\n\nquestion")
        self.assertEqual(examples[0]["raw_question"], "question")
        self.assertEqual(examples[0]["source_id"], "example-1")

    def test_fewshot_predict_uses_stable_input_seed(self) -> None:
        input_text = "same input"
        expected = int.from_bytes(
            hashlib.sha256(input_text.encode()).digest()[:8], "big"
        )
        memory = FewShotMemory(lambda _: '{"final_answer": "ok"}')
        memory._format_examples_section = Mock(return_value="")

        memory.predict(input_text)

        self.assertEqual(_seed_for_input(input_text), expected)
        memory._format_examples_section.assert_called_once_with(seed=expected)


class HeldOutIsolationTests(unittest.TestCase):
    def test_default_evolution_never_runs_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            with (
                patch.object(meta_harness, "EVOLVE_DIR", Path(tmp)),
                patch.object(meta_harness, "AGENTS_DIR", Path(tmp) / "agents"),
                patch.object(
                    meta_harness,
                    "run_benchmark",
                    side_effect=lambda args: calls.append(args),
                ),
                redirect_stdout(io.StringIO()),
            ):
                meta_harness.run_evolve(_args())

        self.assertFalse(any("--test" in call for call in calls))

    def test_finalization_prints_test_results_and_blocks_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "logs" / "test-run"
            run_dir.mkdir(parents=True)
            (run_dir / "frontier_val.json").write_text(
                json.dumps({"_pareto": [{"system": "candidate", "val_accuracy": 75.0}]})
            )
            calls = []

            def fake_benchmark(args):
                calls.append(args)
                if "--memory" in args:
                    system = args[args.index("--memory") + 1]
                    for dataset in ("USPTO", "Symptom2Disease", "LawBench"):
                        result_dir = benchmark_run_dir(
                            meta_harness.RESULTS_DIR,
                            dataset,
                            system,
                            "gpt-6-luna",
                            42,
                        )
                        result_dir.mkdir(parents=True, exist_ok=True)
                        (result_dir / "test.json").write_text('{"accuracy": 1.0}')
                stdout = (
                    "held-out test table" if args == ["--results", "--test"] else ""
                )
                return subprocess.CompletedProcess(args, 0, stdout, "")

            output = io.StringIO()
            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "AGENTS_DIR", root / "agents"),
                patch.object(meta_harness, "run_benchmark", fake_benchmark),
                redirect_stdout(output),
            ):
                meta_harness.run_evolve(_args(test=True))
                with self.assertRaises(SystemExit):
                    meta_harness.run_evolve(_args())

            state = json.loads((run_dir / "finalized.json").read_text())

        self.assertEqual(state["status"], "complete")
        self.assertIn(["--memory", "candidate", "--test"], calls)
        self.assertIn(["--results", "--test"], calls)
        self.assertIn("held-out test table", output.getvalue())

    def test_benchmark_test_results_are_run_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "LOGS_DIR", root / "logs" / "run"),
                patch.object(
                    meta_harness, "RESULTS_DIR", root / "logs" / "run" / "results"
                ),
                patch.object(
                    meta_harness,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run_cmd,
            ):
                meta_harness.run_benchmark(["--results", "--test"])

            command = run_cmd.call_args.args[0]

        results_flag = command.index("--results-dir")
        self.assertEqual(command[results_flag + 1], str(root / "logs/run/results"))
        self.assertEqual(command[-2:], ["--results", "--test"])

    def test_incomplete_finalization_allows_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "logs" / "test-run"
            run_dir.mkdir(parents=True)
            (run_dir / "frontier_val.json").write_text('{"_pareto": []}')

            def corrupt_benchmark(args):
                if "--memory" in args:
                    system = args[args.index("--memory") + 1]
                    for dataset in ("USPTO", "Symptom2Disease", "LawBench"):
                        result_dir = benchmark_run_dir(
                            meta_harness.RESULTS_DIR,
                            dataset,
                            system,
                            "gpt-6-luna",
                            42,
                        )
                        result_dir.mkdir(parents=True, exist_ok=True)
                        (result_dir / "test.json").write_text("{}")
                return subprocess.CompletedProcess(args, 0, "", "")

            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "AGENTS_DIR", root / "agents"),
                patch.object(meta_harness, "run_benchmark", corrupt_benchmark),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    meta_harness.run_evolve(_args(test=True))
                meta_harness.run_evolve(_args())

            state = json.loads((run_dir / "finalized.json").read_text())

        self.assertEqual(state["status"], "in_progress")


class EvolutionSummaryTests(unittest.TestCase):
    def test_delta_uses_frontier_from_start_of_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "evolution_summary.jsonl"
            with patch.object(meta_harness, "EVOLUTION_SUMMARY", summary):
                meta_harness.update_evolution_summary(
                    iteration=1,
                    candidates=[{"name": "candidate", "axis": "retrieval"}],
                    val_scores={"candidate": 90.0},
                    best_val=80.0,
                )

            row = json.loads(summary.read_text())

        self.assertEqual(row["delta"], 10.0)
        self.assertEqual(row["outcome"], "90.0% (+10.0)")


class DatasetBlindOptimizationTests(unittest.TestCase):
    def test_blind_view_exposes_only_aggregate_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agents"
            run_dir = root / "logs" / "blind-run"
            agents.mkdir()
            run_dir.mkdir(parents=True)
            for filename in meta_harness.BASELINE_FILES:
                (agents / filename).write_text("# baseline\n", encoding="utf-8")
            for filename in ("__init__.py", "llm.py", "memory_system.py"):
                (root / filename).write_text("# interface\n", encoding="utf-8")
            (run_dir / "frontier_val.json").write_text(
                json.dumps(
                    {
                        "SecretDataset": {"best_system": "base", "accuracy": 75},
                        "_pareto": [
                            {"system": "base", "val_accuracy": 75, "ctx_len": 10}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "evolution_summary.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "system": "prior",
                        "avg_val": 70,
                        "hypothesis": "SecretDataset-specific observation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (agents / "prior.py").write_text("# prior\n", encoding="utf-8")
            trace_dir = run_dir / "SecretDataset" / "base" / "model"
            trace_dir.mkdir(parents=True)
            (trace_dir / "log.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "meta",
                                "dataset": "SecretDataset",
                                "memory": "base",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "val_step",
                                "input": "a medical-looking problem",
                                "pred": "answer-a",
                                "tgt": "answer-b",
                                "ok": False,
                                "prompt_len": 30,
                                "context_len": 10,
                                "prompt_text": "context plus a medical-looking problem",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "AGENTS_DIR", agents),
                patch.object(meta_harness, "LOGS_DIR", run_dir),
                patch.object(
                    meta_harness, "FRONTIER_VAL", run_dir / "frontier_val.json"
                ),
                patch.object(
                    meta_harness,
                    "EVOLUTION_SUMMARY",
                    run_dir / "evolution_summary.jsonl",
                ),
                patch.object(
                    meta_harness, "PROPOSER_VIEW", run_dir / "proposer_view"
                ),
            ):
                view_dir, pending = meta_harness.prepare_dataset_blind_view(3)

            state_text = (view_dir / "optimizer_state.json").read_text(
                encoding="utf-8"
            )
            state = json.loads(state_text)
            traces_text = (view_dir / "anonymous_traces.jsonl").read_text(
                encoding="utf-8"
            )
            prior_copied = (view_dir / "agents" / "prior.py").exists()
            pending_exists = pending.exists()

        self.assertEqual(state["task_count"], 3)
        self.assertEqual(state["pareto"][0]["avg_val"], 75)
        self.assertNotIn("SecretDataset", state_text)
        self.assertNotIn("hypothesis", state["history"][0])
        self.assertIn("a medical-looking problem", traces_text)
        self.assertNotIn("SecretDataset", traces_text)
        self.assertNotIn('"dataset"', traces_text)
        self.assertTrue(prior_copied)
        self.assertFalse(pending_exists)

    def test_blind_prompt_does_not_point_to_raw_run_state(self) -> None:
        prompt = meta_harness.render_task_prompt(
            2, 3, dataset_blind=True, view_dir=Path("view")
        )

        self.assertIn("anonymous tasks", prompt)
        self.assertIn("optimizer_state.json", prompt)
        self.assertIn("anonymous_traces.jsonl", prompt)
        self.assertNotIn("frontier_val.json", prompt)
        self.assertNotIn("config.yaml", prompt)


if __name__ == "__main__":
    unittest.main()
