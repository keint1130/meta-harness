"""Small Meta-Harness outer loop for math search; final test is intentionally excluded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inner_loop import evaluate, load_candidate
from splits import ROOT, load_config

TEXT_EXAMPLE = ROOT.parent / "text_classification"
sys.path.insert(0, str(TEXT_EXAMPLE))
import codex_wrapper  # noqa: E402


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("global", "task-specific"), required=True)
    parser.add_argument("--domain", choices=config["domains"])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if args.mode == "task-specific" and not args.domain:
        parser.error("--domain is required for task-specific mode")
    if args.mode == "global" and args.domain:
        parser.error("--domain is not valid for global mode")

    domain = args.domain if args.mode == "task-specific" else None
    run_dir = ROOT / "logs" / args.run_name
    results_dir = run_dir / "results"
    summary = run_dir / "evolution_summary.jsonl"
    pending = run_dir / "pending_eval.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_baseline:
        output = results_dir / "iter0_no_memory_search.json"
        result = evaluate(ROOT / "agents" / "no_memory.py", "search", domain, output)
        append_jsonl(summary, {"iteration": 0, **{k: result[k] for k in ("candidate", "domain", "accuracy", "macro_accuracy")}})

    for iteration in range(1, args.iterations + 1):
        history = summary.read_text(encoding="utf-8") if summary.exists() else ""
        prompt = (
            f"Run iteration {iteration} of {args.mode} math harness evolution"
            f"{f' for {domain}' if domain else ' across four balanced domains'}.\n"
            f"Write pending_eval.json to `{pending}`.\n"
            f"Search-only history:\n{history[-12000:]}"
        )
        proposer = config["proposer"]
        session = codex_wrapper.run(
            prompt=prompt,
            model=proposer["model"],
            effort=proposer["reasoning_effort"],
            cwd=str(ROOT),
            skills=[str(ROOT / ".claude" / "skills" / "meta-harness")],
            log_dir=str(run_dir / "codex_sessions"),
            name=f"iter{iteration}",
            timeout_seconds=2400,
        )
        if session.exit_code or not pending.exists():
            raise RuntimeError(f"Proposer failed in iteration {iteration}: {session.stderr}")
        candidates = json.loads(pending.read_text(encoding="utf-8"))["candidates"]
        if len(candidates) != proposer["candidates_per_iteration"]:
            raise ValueError(f"Expected {proposer['candidates_per_iteration']} candidates")
        for candidate in candidates:
            path = ROOT / candidate["file"]
            load_candidate(path)
            output = results_dir / f"iter{iteration}_{candidate['name']}_search.json"
            result = evaluate(path, "search", domain, output)
            append_jsonl(
                summary,
                {
                    "iteration": iteration,
                    "hypothesis": candidate.get("hypothesis", ""),
                    "components": candidate.get("components", []),
                    **{k: result[k] for k in ("candidate", "domain", "accuracy", "macro_accuracy")},
                },
            )
        pending.unlink()


if __name__ == "__main__":
    main()

