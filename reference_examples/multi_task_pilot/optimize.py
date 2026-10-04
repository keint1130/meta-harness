from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from multi_task_pilot.baseline import load_harness_spec
from multi_task_pilot.contracts import TaskFamily
from multi_task_pilot.datasets import load_cases


ROOT = Path(__file__).resolve().parent
TEXT_CLASSIFICATION = ROOT.parent / "text_classification"
sys.path.insert(0, str(TEXT_CLASSIFICATION))
import codex_wrapper  # noqa: E402


def load_run_records(run_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run_name in run_names:
        path = ROOT / "logs" / run_name / "results.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            by_family.setdefault(record["family"], []).append(record)
    # A later run replaces an earlier complete family (used for Level-5 math).
    result: dict[str, list[dict[str, Any]]] = {}
    for family in TaskFamily:
        candidates = [
            (name, [r for r in records if r["family"] == family.value])
            for name in run_names
            for records in [[
                json.loads(line)
                for line in (ROOT / "logs" / name / "results.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]]
        ]
        nonempty = [records for _, records in candidates if records]
        if nonempty:
            result[family.value] = nonempty[-1]
    return result


def build_report(records: dict[str, list[dict[str, Any]]], track: str) -> dict[str, Any]:
    families = [track] if track != "global" else [f.value for f in TaskFamily]
    report: dict[str, Any] = {"track": track, "families": {}}
    prepared = ROOT / "data" / "prepared"
    for family_name in families:
        family = TaskFamily(family_name)
        cases = {c.task.id: c for c in load_cases(prepared, family, "dev")}
        family_records = records[family_name]
        failures = []
        for record in family_records:
            if record["score"]["value"]:
                continue
            case = cases.get(record["task_id"])
            if family is TaskFamily.CODING:
                reference: Any = "Hidden tests failed; raw tests are evaluator-only."
            else:
                reference = case.reference if case else None
            tool_calls = []
            for call in record["tool_calls"]:
                arguments = dict(call.get("arguments", {}))
                if "source" in arguments:
                    arguments["source"] = arguments["source"][:1500]
                tool_calls.append(
                    {
                        "name": call.get("name"),
                        "arguments": arguments,
                        "output": str(call.get("output", ""))[-1500:],
                        "duration_seconds": call.get("duration_seconds"),
                    }
                )
            failures.append(
                {
                    "task_id": record["task_id"],
                    "prompt": case.task.prompt if case else "",
                    "prediction": record["answer"][-8000:],
                    "reference": reference,
                    "score_detail": record["score"]["detail"],
                    "tool_calls": tool_calls,
                    "error": record["error"],
                }
            )
        report["families"][family_name] = {
            "accuracy": sum(r["score"]["value"] for r in family_records)
            / len(family_records),
            "failures": failures,
        }
    return report


def track_score(summary: dict[str, Any], track: str) -> float:
    if track == "global":
        return float(summary["macro_average"])
    return float(summary["family_accuracy"][track])


def propose(
    *, track: str, iteration: int, frontier: Path, report: Path,
    pending: Path, run_root: Path, config: dict[str, Any]
) -> None:
    families = "coding, retrieval, and math" if track == "global" else track
    prompt = f"""You are optimizing a constrained agent harness for the {track} track.
The evaluation families are: {families}.

Read these files:
- current frontier: {frontier.as_posix()}
- development failure report: {report.as_posix()}
- schema/loader: {(ROOT / 'multi_task_pilot/baseline.py').as_posix()}

Create exactly one candidate JSON at {pending.as_posix()}.
It may contain only: name, description, system_prompt, family_instructions.
Do not change model, reasoning effort, LLM/tool budgets, datasets, evaluators, or any
other repository file. Improve the instructions based on concrete development
failures without including answers, task IDs, or problem-specific memorization.
For a task-specific track, include at most that family's instruction. The candidate
must generalize to held-out examples. This is optimization iteration {iteration}.
"""
    proposer = config["proposer"]
    result = codex_wrapper.run(
        prompt=prompt,
        model=proposer["model"],
        effort=proposer.get("reasoning_effort"),
        cwd=str(ROOT),
        log_dir=str(run_root / "codex_sessions"),
        name=f"{track}-iter{iteration}",
        timeout_seconds=int(config["optimization"].get("proposer_timeout_seconds", 2400)),
    )
    if result.exit_code:
        raise RuntimeError(f"Codex proposer failed: {result.stderr[:500]}")
    if not pending.is_file():
        raise RuntimeError("Codex proposer did not create pending candidate")


def evaluate(spec: Path, track: str, run_name: str) -> dict[str, Any]:
    families = [f.value for f in TaskFamily] if track == "global" else [track]
    command = [
        "uv", "run", "--env-file", "../../.env", "python", "run_baseline.py",
        "--run-name", run_name, "--harness-spec", str(spec), "--families", *families,
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Candidate evaluation failed with {completed.returncode}")
    return json.loads((ROOT / "logs" / run_name / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Equal-budget constrained harness optimization.")
    parser.add_argument("--tracks", nargs="+", choices=["global", *[f.value for f in TaskFamily]], default=["coding", "retrieval", "math", "global"])
    parser.add_argument("--baseline-runs", nargs="+", default=["baseline-dev", "baseline-math-level5"])
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    iterations = args.iterations or int(config["optimization"]["iterations"])
    records = load_run_records(args.baseline_runs)
    run_name = args.run_name or datetime.now().strftime("evolution-%Y%m%d-%H%M%S")
    run_root = ROOT / "evolution" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "config.yaml", run_root / "config.yaml")
    events = run_root / "evolution.jsonl"

    for track in args.tracks:
        track_dir = run_root / track
        track_dir.mkdir()
        frontier = track_dir / "frontier.json"
        shutil.copy2(ROOT / "harness_specs" / "baseline.json", frontier)
        report_path = track_dir / "report-0.json"
        report_path.write_text(json.dumps(build_report(records, track), indent=2, ensure_ascii=False), encoding="utf-8")
        baseline_scores = [records[f.value] for f in TaskFamily] if track == "global" else [records[track]]
        means = [sum(r["score"]["value"] for r in rows) / len(rows) for rows in baseline_scores]
        best = sum(means) / len(means)
        for iteration in range(1, iterations + 1):
            pending = track_dir / f"candidate-{iteration}.json"
            propose(track=track, iteration=iteration, frontier=frontier, report=report_path, pending=pending, run_root=run_root, config=config)
            spec = load_harness_spec(pending)
            if track != "global" and set(spec.get("family_instructions", {})) - {track}:
                raise ValueError(f"{track} candidate contains other family instructions")
            eval_name = f"{run_name}-{track}-iter{iteration}"
            summary = evaluate(pending, track, eval_name)
            score = track_score(summary, track)
            previous_best = best
            accepted = score > best
            if accepted:
                best = score
                shutil.copy2(pending, frontier)
            event = {"track": track, "iteration": iteration, "candidate": spec["name"], "score": score, "previous_best": previous_best, "accepted": accepted, "evaluation_run": eval_name}
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            new_records = load_run_records([eval_name])
            report_path = track_dir / f"report-{iteration}.json"
            report_path.write_text(json.dumps(build_report(new_records, track), indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
