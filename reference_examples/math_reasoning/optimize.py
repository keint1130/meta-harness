"""Resumable Meta-Harness outer loop for math search.

Only the search split is available to the proposer. Development and final-test
evaluation are intentionally separate operations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from frontier import pareto_frontier, result_point
from inner_loop import evaluate, math_repo_path
from splits import ROOT, load_config

TEXT_EXAMPLE = ROOT.parent / "text_classification"


def load_codex_wrapper():
    path = TEXT_EXAMPLE / "codex_wrapper.py"
    spec = importlib.util.spec_from_file_location("math_codex_wrapper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Codex wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codex_wrapper = load_codex_wrapper()


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def git_state(path: Path) -> dict:
    """Capture reproducibility metadata without recording credentials."""
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain=v1")
    diff = run("diff", "--binary", "--no-ext-diff")
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def initialize_run_manifest(
    run_dir: Path,
    config: dict,
    *,
    run_name: str,
    mode: str,
    domain: str | None,
    candidates_per_iteration: int,
    requested_iterations: int,
    baseline_sources: list[dict],
) -> None:
    """Write immutable run settings and append invocation-level targets."""
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    invariant = {
        "run_name": run_name,
        "mode": mode,
        "domain": domain or "global",
        "candidates_per_iteration": candidates_per_iteration,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "baseline_sources": baseline_sources,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {key: manifest.get(key) for key in invariant}
        if actual != invariant:
            raise ValueError(
                f"Run manifest mismatch; use a new run name. "
                f"expected={invariant}, existing={actual}"
            )
    else:
        math_repo = math_repo_path(config)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **invariant,
            "config": config,
            "authentication": {
                "solver_and_judge": "OPENAI_API_KEY",
                "proposer": os.environ.get(
                    "MOHARNESS_PROPOSER_AUTH_METHOD", "unknown"
                ),
                "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
            },
            "code": {
                "meta_harness": git_state(ROOT.parents[1]),
                "math_frontier_eval": git_state(math_repo),
                "integration_repo": git_state(ROOT.parents[2]),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_jsonl(
        run_dir / "run_invocations.jsonl",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "requested_iterations": requested_iterations,
            "candidates_per_iteration": candidates_per_iteration,
        },
    )


def result_summary(result: dict, iteration: int, **extra) -> dict:
    summary = {
        "iteration": iteration,
        **extra,
        **{
            key: result[key]
            for key in (
                "candidate",
                "domain",
                "accuracy",
                "macro_accuracy",
                "solver_input_tokens",
                "solver_output_tokens",
                "judge_input_tokens",
                "judge_output_tokens",
                "empty_responses",
            )
            if key in result
        },
    }
    summary["search_accuracy"] = result.get("accuracy")
    summary["search_macro_accuracy"] = result.get("macro_accuracy")
    return summary


def validate_result(result: dict, domain: str | None, source: Path) -> None:
    expected_domain = domain or "global"
    if result.get("split") != "search":
        raise ValueError(f"Baseline must use split=search: {source}")
    if result.get("domain") != expected_domain:
        raise ValueError(
            f"Baseline domain mismatch: expected {expected_domain}, "
            f"got {result.get('domain')}: {source}"
        )
    if not isinstance(result.get("details"), list) or not result["details"]:
        raise ValueError(f"Baseline has no problem details: {source}")


def configured_baseline_sources(config: dict, domain: str | None) -> list[dict]:
    configured = config.get("baseline", {}).get("search_results", {})
    domains = [domain] if domain else config["domains"]
    sources = []
    for current_domain in domains:
        raw = configured.get(current_domain)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"No configured search baseline for {current_domain}")
        path = (ROOT / raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured search baseline does not exist for {current_domain}: {path}"
            )
        sources.append(
            {
                "domain": current_domain,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sources


def validate_baseline_config(result: dict, config: dict, source: Path) -> str:
    expected = {
        "experiment": config["experiment"],
        "judge": config.get("judge", {}),
        "memory": config.get("memory", {}),
        "splits": config["splits"],
    }
    recorded = result.get("evaluation_config")
    if recorded is None:
        if not config["baseline"].get("allow_legacy_without_evaluation_config", False):
            raise ValueError(
                f"Baseline lacks evaluation_config and legacy use is disabled: {source}"
            )
        if result.get("model") != config["experiment"]["model"]:
            raise ValueError(f"Legacy baseline model mismatch: {source}")
        return "legacy_partial"
    if recorded != expected:
        raise ValueError(
            f"Baseline evaluation_config differs from current config: {source}"
        )
    return "verified"


def seed_baseline(
    sources: list[dict], destination: Path, domain: str | None, config: dict
) -> dict:
    loaded = []
    for metadata in sources:
        source = Path(metadata["path"])
        result = json.loads(source.read_text(encoding="utf-8"))
        validate_result(result, metadata["domain"], source)
        metadata["compatibility"] = validate_baseline_config(result, config, source)
        expected_count = int(config["splits"]["search"])
        if len(result["details"]) != expected_count:
            raise ValueError(
                f"Baseline problem count mismatch: expected {expected_count}, "
                f"got {len(result['details'])}: {source}"
            )
        loaded.append(result)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if domain:
        combined = dict(loaded[0])
    else:
        details = [detail for item in loaded for detail in item["details"]]
        domain_accuracy = {item["domain"]: item["accuracy"] for item in loaded}
        combined = {
            "candidate": "no_memory",
            "split": "search",
            "domain": "global",
            "n_problems": len(details),
            "accuracy": sum(
                item["accuracy"] * len(item["details"]) for item in loaded
            ) / len(details),
            "macro_accuracy": sum(domain_accuracy.values()) / len(domain_accuracy),
            "domain_accuracy": domain_accuracy,
            "solver_input_tokens": sum(item.get("solver_input_tokens", 0) for item in loaded),
            "solver_output_tokens": sum(item.get("solver_output_tokens", 0) for item in loaded),
            "judge_input_tokens": sum(item.get("judge_input_tokens", 0) for item in loaded),
            "judge_output_tokens": sum(item.get("judge_output_tokens", 0) for item in loaded),
            "total_input_tokens": sum(item.get("total_input_tokens", 0) for item in loaded),
            "total_output_tokens": sum(item.get("total_output_tokens", 0) for item in loaded),
            "elapsed_seconds": sum(item.get("elapsed_seconds", 0) for item in loaded),
            "empty_responses": sum(item.get("empty_responses", 0) for item in loaded),
            "details": details,
        }
    combined["baseline_sources"] = sources
    destination.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def candidate_path(candidate: dict, candidate_dir: Path) -> Path:
    raw = candidate.get("file")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Every candidate must provide a non-empty file path")
    path = (ROOT / raw).resolve()
    expected_root = candidate_dir.resolve()
    if not path.is_relative_to(expected_root):
        raise ValueError(f"Candidate must be inside {expected_root}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    return path


def validate_candidate_source(path: Path) -> None:
    """Validate candidate structure without importing runtime dependencies.

    Candidate modules may import ``math_retriever`` from the sibling evaluation
    repository. That path is intentionally installed by ``evaluate`` immediately
    before the real import, so an eager optimizer-side import would reject valid
    retrieval candidates.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not any(
        isinstance(node, ast.ClassDef) and node.name == "Candidate"
        for node in tree.body
    ):
        raise TypeError(f"{path} must expose class Candidate")


def completed_names(records: list[dict], iteration: int) -> set[str]:
    return {
        str(record["candidate"])
        for record in records
        if record.get("iteration") == iteration and record.get("candidate")
    }


def ensure_new_or_resumable(run_dir: Path, resume: bool) -> None:
    artifacts = (
        run_dir / "evolution_summary.jsonl",
        run_dir / "pending_eval.json",
        run_dir / "results",
        run_dir / "codex_sessions",
    )
    if run_dir.exists() and any(path.exists() for path in artifacts) and not resume:
        raise FileExistsError(
            f"Run already contains state: {run_dir}. Pass --resume or use a new run name."
        )


def ensure_search_mutable(run_dir: Path) -> None:
    """Do not adapt search after final-test access."""
    if (run_dir / "finalized.json").exists():
        raise ValueError(f"Run is frozen for final testing: {run_dir}")


def record_candidate_path(record: dict, run_dir: Path) -> Path:
    name = record.get("candidate")
    if record.get("iteration") == 0 and name == "no_memory":
        return ROOT / "agents" / "no_memory.py"
    raw = record.get("candidate_file")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Candidate {name!r} has no candidate_file")
    path = (ROOT / raw).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Candidate source is outside its run: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Candidate source not found: {path}")
    return path


def update_dev_frontier(
    run_dir: Path, summary: Path, mode: str, domain: str | None
) -> list[dict]:
    """Backfill missing dev evaluations and publish aggregate-only feedback."""
    records = read_jsonl(summary)
    changed = False
    for position, record in enumerate(records, 1):
        if record.get("dev_accuracy") is not None:
            continue
        name = record.get("candidate")
        if not isinstance(name, str) or not name:
            continue
        candidate = record_candidate_path(record, run_dir)
        output = run_dir / "dev" / f"{name}.json"
        if output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            if result.get("split") != "dev" or result.get("domain") != (domain or "global"):
                raise ValueError(f"Existing dev result has wrong scope: {output}")
            print(f"[SKIP] existing dev result: {name}", flush=True)
        else:
            print(f"[DEV] candidate {position}/{len(records)}: {name}", flush=True)
            result = evaluate(candidate, "dev", domain, output)
        record.update(
            {
                "dev_accuracy": result["accuracy"],
                "dev_macro_accuracy": result["macro_accuracy"],
                "dev_solver_input_tokens": result["solver_input_tokens"],
                "dev_solver_output_tokens": result["solver_output_tokens"],
                "dev_empty_responses": result["empty_responses"],
                "dev_result": str(output.relative_to(ROOT)),
            }
        )
        changed = True
        write_jsonl(summary, records)

    points = []
    for record in records:
        dev_result = record.get("dev_result")
        if not dev_result:
            continue
        result = json.loads((ROOT / dev_result).read_text(encoding="utf-8"))
        points.append(result_point(result, mode, record_candidate_path(record, run_dir)))
    if points:
        pareto = pareto_frontier(points)
        best = min(
            points,
            key=lambda row: (
                -row["accuracy"],
                row["solver_tokens_per_problem"],
                row["empty_responses"],
            ),
        )
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
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
        (run_dir / "frontier_val.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    if changed:
        print(f"[FRONTIER] updated {run_dir / 'frontier_val.json'}", flush=True)
    return records


def prepare_proposer_view(
    run_dir: Path,
    candidate_dir: Path,
    records: list[dict],
    *,
    mode: str,
    domain: str | None,
) -> tuple[Path, Path]:
    """Create the only workspace visible to the proposer.

    Problem IDs, benchmark domain/subcategory labels, per-domain global scores,
    and evaluator configuration are deliberately not copied.
    """
    view = run_dir / "proposer_view" / "math_reasoning"
    agents = view / "agents"
    if view.exists():
        shutil.rmtree(view)
    agents.mkdir(parents=True)

    for source in sorted((ROOT / "agents").glob("*.py")):
        shutil.copy2(source, agents / source.name)
    for source in sorted((run_dir / "agents").glob("iter*/*.py")):
        target = agents / source.name
        if not target.exists() or target.read_bytes() == source.read_bytes():
            shutil.copy2(source, target)

    safe_history = []
    for row in records:
        safe_history.append(
            {
                key: row[key]
                for key in (
                    "iteration",
                    "candidate",
                    "accuracy",
                    "macro_accuracy",
                    "solver_input_tokens",
                    "solver_output_tokens",
                    "empty_responses",
                    "search_accuracy",
                    "search_macro_accuracy",
                    "dev_accuracy",
                    "dev_macro_accuracy",
                    "dev_solver_input_tokens",
                    "dev_solver_output_tokens",
                    "dev_empty_responses",
                    "hypothesis",
                    "base_system",
                    "components",
                )
                if key in row
            }
        )
    state = {
        "mode": "task_specific" if mode == "task-specific" else "domain_blind_global",
        "visible_task_context": {"domain": domain} if domain else {},
        "objective": "search accuracy" if domain else "unweighted macro-average accuracy",
        "history": safe_history,
    }
    (view / "optimizer_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    traces = []
    for result_path in sorted((run_dir / "results").glob("*_search.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for detail in result.get("details", []):
            trace = {
                "candidate": result.get("candidate"),
                "problem": detail.get("problem", ""),
                "prompt": detail.get("prompt", ""),
                "response": detail.get("response", ""),
                "ground_truth": detail.get("groundtruth", ""),
                "score": detail.get("score", 0.0),
                "deterministic_score": detail.get("deterministic_score", 0.0),
                "used_judge": detail.get("judge_score") is not None,
            }
            encoded = json.dumps(trace, sort_keys=True, ensure_ascii=False)
            trace["trace_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:12]
            traces.append(trace)
    traces.sort(key=lambda row: row["trace_id"])
    (view / "anonymous_traces.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces),
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "math_harness.py", view / "math_harness.py")
    pending = view / "pending_eval.json"
    return view, pending


def install_proposer_candidates(
    view: Path, view_pending: Path, candidate_dir: Path, pending: Path
) -> dict:
    if not view_pending.is_file():
        raise FileNotFoundError(f"Proposer did not write {view_pending}")
    manifest = json.loads(view_pending.read_text(encoding="utf-8"))
    for candidate in manifest.get("candidates", []):
        name = candidate.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid candidate name: {name!r}")
        source = view / "agents" / f"{name}.py"
        if not source.is_file():
            raise FileNotFoundError(f"Candidate file not found in proposer view: {source}")
        target = candidate_dir / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Candidate would overwrite different source: {target}")
        shutil.copy2(source, target)
        candidate["file"] = target.relative_to(ROOT).as_posix()
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("global", "task-specific"), required=True)
    parser.add_argument("--domain", choices=config["domains"])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--candidates-per-iteration",
        type=int,
        default=int(config["proposer"]["candidates_per_iteration"]),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "task-specific" and not args.domain:
        parser.error("--domain is required for task-specific mode")
    if args.mode == "global" and args.domain:
        parser.error("--domain is not valid for global mode")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.candidates_per_iteration < 1:
        parser.error("--candidates-per-iteration must be at least 1")

    domain = args.domain if args.mode == "task-specific" else None
    run_dir = ROOT / "logs" / args.run_name
    ensure_search_mutable(run_dir)
    ensure_new_or_resumable(run_dir, args.resume)
    results_dir = run_dir / "results"
    summary = run_dir / "evolution_summary.jsonl"
    pending = run_dir / "pending_eval.json"
    baseline_sources = configured_baseline_sources(config, domain)
    run_dir.mkdir(parents=True, exist_ok=True)
    initialize_run_manifest(
        run_dir,
        config,
        run_name=args.run_name,
        mode=args.mode,
        domain=domain,
        candidates_per_iteration=args.candidates_per_iteration,
        requested_iterations=args.iterations,
        baseline_sources=baseline_sources,
    )
    records = read_jsonl(summary)

    baseline_output = results_dir / "iter0_no_memory_search.json"
    baseline_records = [record for record in records if record.get("iteration") == 0]
    if baseline_records:
        print("[SKIP] baseline already recorded", flush=True)
    else:
        result = seed_baseline(baseline_sources, baseline_output, domain, config)
        append_jsonl(
            summary,
            result_summary(
                result,
                0,
                baseline_sources=baseline_sources,
            ),
        )
        print("[SEED] baseline from config", flush=True)

    # Validation is the outer-loop objective. Only these aggregate fields are
    # exposed to the proposer; detailed dev result files stay outside its view.
    records = update_dev_frontier(run_dir, summary, args.mode, domain)

    for iteration in range(1, args.iterations + 1):
        records = read_jsonl(summary)
        done = completed_names(records, iteration)
        if len(done) > args.candidates_per_iteration:
            raise ValueError(
                f"Iteration {iteration} has more completed candidates than requested"
            )
        if len(done) == args.candidates_per_iteration:
            missing_outputs = [
                name
                for name in done
                if not (results_dir / f"iter{iteration}_{name}_search.json").is_file()
            ]
            if missing_outputs:
                raise ValueError(
                    f"Iteration {iteration} summary has missing result files: "
                    f"{sorted(missing_outputs)}"
                )
            if pending.exists():
                pending_iteration = json.loads(
                    pending.read_text(encoding="utf-8")
                ).get("iteration")
                if pending_iteration == iteration:
                    pending.unlink()
            print(f"[SKIP] iteration {iteration} already complete", flush=True)
            continue
        candidate_dir = run_dir / "agents" / f"iter{iteration}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if pending.exists():
            manifest = json.loads(pending.read_text(encoding="utf-8"))
            if manifest.get("iteration") != iteration:
                raise ValueError(
                    f"Stale pending manifest for iteration {manifest.get('iteration')}: {pending}"
                )
            print(f"[RESUME] candidate manifest for iteration {iteration}", flush=True)
        else:
            view, view_pending = prepare_proposer_view(
                run_dir,
                candidate_dir,
                records,
                mode=args.mode,
                domain=domain,
            )
            prompt = (
                f"Run iteration {iteration} of {args.mode} math harness evolution.\n"
                f"Implement exactly {args.candidates_per_iteration} candidates.\n"
                "Work only inside the supplied isolated directory. Do not inspect parent "
                "directories or external benchmark data/config/logs.\n"
                "Read optimizer_state.json and anonymous_traces.jsonl. Domain and "
                "subcategory labels absent from those files are intentionally hidden.\n"
                "Write candidates to agents/<name>.py and pending_eval.json in the "
                f"working directory with iteration={iteration}."
            )
            proposer = config["proposer"]
            print(f"[PROPOSE] iteration {iteration}", flush=True)
            session = codex_wrapper.run(
                prompt=prompt,
                model=proposer["model"],
                effort=proposer["reasoning_effort"],
                cwd=str(view),
                skills=[str(ROOT / ".claude" / "skills" / "meta-harness-blind")],
                log_dir=str(run_dir / "codex_sessions"),
                name=f"iter{iteration}",
                timeout_seconds=2400,
            )
            if session.exit_code or not view_pending.exists():
                raise RuntimeError(
                    f"Proposer failed in iteration {iteration}: {session.stderr}"
                )
            manifest = install_proposer_candidates(
                view, view_pending, candidate_dir, pending
            )

        candidates = manifest.get("candidates", [])
        if len(candidates) != args.candidates_per_iteration:
            raise ValueError(
                f"Expected {args.candidates_per_iteration} candidates, got {len(candidates)}"
            )
        names = [candidate.get("name") for candidate in candidates]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Every candidate must have a non-empty name")
        if len(names) != len(set(names)):
            raise ValueError("Candidate names must be unique within an iteration")

        done = completed_names(read_jsonl(summary), iteration)
        for position, candidate in enumerate(candidates, start=1):
            name = candidate["name"]
            path = candidate_path(candidate, candidate_dir)
            validate_candidate_source(path)
            output = results_dir / f"iter{iteration}_{name}_search.json"
            if name in done and output.exists():
                print(f"[SKIP] iteration {iteration} candidate {name}", flush=True)
                continue
            if name in done or output.exists():
                raise ValueError(
                    f"Inconsistent resume state for iteration {iteration} candidate {name}"
                )
            print(
                f"[EVAL] iteration {iteration} candidate {position}/{len(candidates)}: {name}",
                flush=True,
            )
            result = evaluate(path, "search", domain, output)
            append_jsonl(
                summary,
                result_summary(
                    result,
                    iteration,
                    hypothesis=candidate.get("hypothesis", ""),
                    base_system=candidate.get("base_system", ""),
                    components=candidate.get("components", []),
                    candidate_file=str(path.relative_to(ROOT)),
                ),
            )
            update_dev_frontier(run_dir, summary, args.mode, domain)
        pending.unlink()
        print(f"[DONE] iteration {iteration}", flush=True)


if __name__ == "__main__":
    main()
