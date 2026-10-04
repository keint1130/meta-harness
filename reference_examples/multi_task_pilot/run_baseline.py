from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from multi_task_pilot.baseline import BaselineHarness, load_harness_spec
from multi_task_pilot.coding_evaluation import evaluate_livecodebench
from multi_task_pilot.contracts import TaskFamily
from multi_task_pilot.datasets import load_cases
from multi_task_pilot.docker_sandbox import DockerSandbox
from multi_task_pilot.evaluation import Score, macro_average, score_math, score_popqa
from multi_task_pilot.tools import FrozenTools, ToolBudgets


ROOT = Path(__file__).resolve().parent


def score_case(case, answer: str, sandbox: DockerSandbox) -> Score:
    if case.task.family is TaskFamily.CODING:
        result = evaluate_livecodebench(case, answer, sandbox)
        return Score(result.passed, float(result.passed), result.detail)
    if case.task.family is TaskFamily.RETRIEVAL:
        return score_popqa(answer, case.reference)
    return score_math(answer, case.reference)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the shared baseline harness.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="Limit each family for a smoke run.")
    parser.add_argument(
        "--families", nargs="+", choices=[f.value for f in TaskFamily],
        default=[f.value for f in TaskFamily],
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--harness-spec", type=Path, default=ROOT / "harness_specs" / "baseline.json"
    )
    parser.add_argument("--task-ids", nargs="+", help="Run only these frozen task IDs.")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    docker_cfg = config["docker"]
    sandbox = DockerSandbox(
        image=docker_cfg["image"],
        timeout_seconds=docker_cfg["timeout_seconds"],
        memory=docker_cfg["memory"],
        cpus=docker_cfg["cpus"],
        pids_limit=docker_cfg["pids_limit"],
    )
    budgets = ToolBudgets(
        python_calls=config["budgets"]["max_python_calls"],
        retrieve_calls=config["budgets"]["max_retrieve_calls"],
        retrieved_documents=config["budgets"]["max_retrieved_documents"],
    )
    harness_spec = load_harness_spec(args.harness_spec)
    harness = BaselineHarness(
        model=config["model"]["name"],
        reasoning_effort=config["model"].get("reasoning_effort"),
        max_llm_calls=config["budgets"]["max_llm_calls"],
        system_prompt=harness_spec["system_prompt"],
        family_instructions=harness_spec.get("family_instructions"),
    )
    run_name = args.run_name or datetime.now().strftime("baseline-%Y%m%d-%H%M%S")
    output_dir = ROOT / "logs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "config.yaml", output_dir / "config.yaml")
    shutil.copytree(ROOT / "manifests", output_dir / "manifests")
    shutil.copy2(args.harness_spec, output_dir / "harness.json")
    scores: dict[TaskFamily, list[Score]] = defaultdict(list)
    totals = defaultdict(int)

    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as log:
        for family_name in args.families:
            family = TaskFamily(family_name)
            cases = load_cases(ROOT / "data" / "prepared", family, args.split)
            if args.task_ids:
                requested = set(args.task_ids)
                cases = [case for case in cases if case.task.id in requested]
            if args.limit is not None:
                cases = cases[: args.limit]
            for index, case in enumerate(cases, start=1):
                tools = FrozenTools(resources=case.resources, sandbox=sandbox, budgets=budgets)
                try:
                    result = harness.solve(case.task, tools)
                    score = score_case(case, result.answer, sandbox)
                    error = None
                except Exception as exc:
                    result = None
                    score = Score(False, 0.0, "runtime_error")
                    error = f"{type(exc).__name__}: {exc}"
                scores[family].append(score)
                if result:
                    for key, value in result.usage.items():
                        totals[key] += value
                record = {
                    "task_id": case.task.id,
                    "family": family.value,
                    "score": dataclasses.asdict(score),
                    "answer": result.answer if result else "",
                    "usage": result.usage if result else {},
                    "tool_calls": [dataclasses.asdict(call) for call in tools.calls],
                    "error": error,
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(f"[{family.value} {index}/{len(cases)}] score={score.value:.0f}")

    family_accuracy = {
        family.value: sum(score.value for score in values) / len(values)
        for family, values in scores.items()
    }
    summary = {
        "run_name": run_name,
        "split": args.split,
        "model": config["model"],
        "harness": harness_spec["name"],
        "family_accuracy": family_accuracy,
        "macro_average": macro_average(scores) if len(scores) == len(TaskFamily) else None,
        "usage": dict(totals),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
