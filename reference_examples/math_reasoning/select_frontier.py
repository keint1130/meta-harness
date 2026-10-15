"""Development selection and one-shot finalization for math harness runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from frontier import (
    objective_accuracy,
    pareto_frontier,
    result_point,
    solver_tokens_per_problem,
)
from inner_loop import evaluate
from optimize import read_jsonl
from splits import ROOT, load_config


def candidate_inventory(run_dir: Path) -> list[dict]:
    """Resolve one source file per search candidate without leaving the run."""
    records = read_jsonl(run_dir / "evolution_summary.jsonl")
    if not records:
        raise ValueError(f"No search history found in {run_dir}")
    inventory = []
    seen = set()
    for record in records:
        name = record.get("candidate")
        if not isinstance(name, str) or not name or name in seen:
            continue
        if record.get("iteration") == 0 and name == "no_memory":
            path = ROOT / "agents" / "no_memory.py"
        else:
            raw = record.get("candidate_file")
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"Candidate {name} has no candidate_file in search history")
            path = (ROOT / raw).resolve()
            if not path.is_relative_to(run_dir.resolve()):
                raise ValueError(f"Candidate source is outside its run: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Candidate source not found: {path}")
        inventory.append({"name": name, "path": path, "iteration": record.get("iteration")})
        seen.add(name)
    return inventory


def load_complete_result(path: Path, split: str, domain: str | None) -> dict | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_domain = domain or "global"
    if result.get("split") != split or result.get("domain") != expected_domain:
        raise ValueError(f"Existing result does not match {split}/{expected_domain}: {path}")
    if not result.get("details"):
        raise ValueError(f"Existing result is incomplete: {path}")
    return result


def run_dev(run_dir: Path, mode: str, domain: str | None) -> dict:
    if (run_dir / "finalized.json").exists():
        raise ValueError("Run is already frozen for final-test evaluation")
    if (run_dir / "pending_eval.json").exists():
        raise ValueError("Search has an unfinished candidate iteration; resume it before dev")
    inventory = candidate_inventory(run_dir)
    dev_dir = run_dir / "dev"
    points = []
    for position, candidate in enumerate(inventory, 1):
        output = dev_dir / f"{candidate['name']}.json"
        result = load_complete_result(output, "dev", domain)
        if result is None:
            print(
                f"[DEV] candidate {position}/{len(inventory)}: {candidate['name']}",
                flush=True,
            )
            result = evaluate(candidate["path"], "dev", domain, output)
        else:
            print(f"[SKIP] existing dev result: {candidate['name']}", flush=True)
        points.append(result_point(result, mode, candidate["path"]))

    pareto = pareto_frontier(points)
    best = min(
        points,
        key=lambda row: (-row["accuracy"], row["solver_tokens_per_problem"], row["empty_responses"]),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "domain": domain or "global",
        "selection_split": "dev",
        "feedback_policy": "aggregate_only",
        "objective": "macro_accuracy" if mode == "global" else "accuracy",
        "cost_metric": "solver_input_plus_output_tokens_per_problem",
        "best_accuracy": best,
        "_pareto": pareto,
        "systems": points,
    }
    path = run_dir / "frontier_val.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[FRONTIER] wrote {path}", flush=True)
    for point in pareto:
        print(
            f"  {point['system']}: accuracy={point['accuracy']:.1%} "
            f"solver_tokens/problem={point['solver_tokens_per_problem']:.0f}",
            flush=True,
        )
    return payload


def run_test(run_dir: Path, mode: str, domain: str | None) -> dict:
    frontier_path = run_dir / "frontier_val.json"
    if not frontier_path.is_file():
        raise ValueError("Run dev selection first; frontier_val.json is missing")
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    if frontier.get("mode") != mode or frontier.get("domain") != (domain or "global"):
        raise ValueError("Frontier mode/domain does not match this finalization request")

    lock_path = run_dir / "finalized.json"
    state = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    if state.get("status") == "complete":
        print("[SKIP] final-test evaluation already complete", flush=True)
        return state

    selected = selected_frontier_systems(frontier)
    if not selected:
        raise ValueError("Frontier contains no systems to evaluate")

    state = {
        "status": "in_progress",
        "started_at": state.get("started_at", datetime.now(timezone.utc).isoformat()),
        "mode": mode,
        "domain": domain or "global",
        "systems": sorted(selected),
    }
    lock_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    test_dir = run_dir / "test"
    summaries = []
    for position, name in enumerate(sorted(selected), 1):
        candidate_path = Path(selected[name]["candidate_path"])
        output = test_dir / f"{name}.json"
        result = load_complete_result(output, "test", domain)
        if result is None:
            print(f"[TEST] candidate {position}/{len(selected)}: {name}", flush=True)
            result = evaluate(candidate_path, "test", domain, output)
        else:
            print(f"[SKIP] existing test result: {name}", flush=True)
        summaries.append(result_point(result, mode, candidate_path))

    state.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": summaries,
        }
    )
    lock_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[DONE] final-test results written under {test_dir}", flush=True)
    return state


def selected_frontier_systems(frontier: dict) -> dict[str, dict]:
    """Return baseline, best-accuracy, and all Pareto systems."""
    selected = {row["system"]: row for row in frontier.get("_pareto", [])}
    best = frontier.get("best_accuracy")
    if best:
        selected[best["system"]] = best
    baseline = next(
        (row for row in frontier.get("systems", []) if row["system"] == "no_memory"),
        None,
    )
    if baseline:
        selected["no_memory"] = baseline
    return selected


def run_smoke(
    run_dir: Path,
    mode: str,
    domain: str | None,
    problem_limit: int = 2,
) -> dict:
    """Exercise finalization on exposed search problems, never final test."""
    if problem_limit < 1:
        raise ValueError("problem_limit must be at least 1")
    frontier_path = run_dir / "frontier_val.json"
    if not frontier_path.is_file():
        raise ValueError("Run dev selection first; frontier_val.json is missing")
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    if frontier.get("mode") != mode or frontier.get("domain") != (domain or "global"):
        raise ValueError("Frontier mode/domain does not match this smoke request")
    selected = selected_frontier_systems(frontier)
    if not selected:
        raise ValueError("Frontier contains no systems to smoke-test")

    output_dir = run_dir / "smoke_finalization"
    summaries = []
    for position, name in enumerate(sorted(selected), 1):
        candidate_path = Path(selected[name]["candidate_path"])
        output = output_dir / f"{name}.json"
        result = load_complete_result(output, "search", domain)
        if result is not None and result.get("n_problems") != problem_limit:
            raise ValueError(
                f"Existing smoke result used {result.get('n_problems')} problems, "
                f"requested {problem_limit}: {output}"
            )
        if result is None:
            print(
                f"[SMOKE] candidate {position}/{len(selected)}: {name} "
                f"({problem_limit} exposed search problems)",
                flush=True,
            )
            result = evaluate(
                candidate_path,
                "search",
                domain,
                output,
                problem_limit=problem_limit,
            )
        else:
            print(f"[SKIP] existing smoke result: {name}", flush=True)
        summaries.append(result_point(result, mode, candidate_path))

    payload = {
        "status": "complete",
        "split": "search",
        "purpose": "finalization_pipeline_smoke_only",
        "problem_limit": problem_limit,
        "mode": mode,
        "domain": domain or "global",
        "systems": summaries,
        "final_test_opened": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] smoke finalization complete; final test remains unopened: {output_dir}",
        flush=True,
    )
    return payload


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("dev", "smoke", "test"), default="dev")
    parser.add_argument("--mode", choices=("global", "task-specific"), required=True)
    parser.add_argument("--domain", choices=config["domains"])
    parser.add_argument("--smoke-problems", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "task-specific" and not args.domain:
        parser.error("--domain is required for task-specific mode")
    if args.mode == "global" and args.domain:
        parser.error("--domain is not valid for global mode")
    run_dir = ROOT / "logs" / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    domain = args.domain if args.mode == "task-specific" else None
    if args.phase == "dev":
        run_dev(run_dir, args.mode, domain)
    elif args.phase == "smoke":
        run_smoke(run_dir, args.mode, domain, args.smoke_problems)
    else:
        run_test(run_dir, args.mode, domain)


if __name__ == "__main__":
    main()
