"""Autonomous evolution loop for memory systems.

Validation-only during evolution. Test evaluation requires explicit finalization.
Uses codex_wrapper + meta-harness skill to propose new memory systems.

    uv run python meta_harness.py --iterations 20 --fresh
    uv run python meta_harness.py --iterations 10 --run-name my-run
    uv run python meta_harness.py --run-name my-run --test
"""

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import codex_wrapper
from benchmark import get_model_short_name, load_results

EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = EVOLVE_DIR / "config.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}

# These are updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
RESULTS_DIR = LOGS_DIR / "results"
FINALIZED = LOGS_DIR / "finalized.json"
PROPOSER_VIEW = LOGS_DIR / "proposer_view"

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

_interrupted = False

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=7200, cwd=None):
    """Wraps subprocess.run; returns CompletedProcess with returncode=124 on timeout."""
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout="", stderr=f"Timed out after {timeout}s"
        )


def run_benchmark(args):
    return run_cmd(
        [
            "uv",
            "run",
            "python",
            "benchmark.py",
            "--logs-dir",
            str(LOGS_DIR),
            "--results-dir",
            str(RESULTS_DIR),
        ]
        + args,
        cwd=str(EVOLVE_DIR),
    )


def _read_evolution_history():
    if not EVOLUTION_SUMMARY.exists():
        return []
    rows = []
    for line in EVOLUTION_SUMMARY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def prepare_dataset_blind_view(num_datasets):
    """Build an isolated, aggregate-only workspace for the proposer."""
    package_dir = PROPOSER_VIEW / "text_classification"
    view_agents = package_dir / "agents"
    view_agents.mkdir(parents=True, exist_ok=True)

    history = _read_evolution_history()
    prior_candidates = {
        row.get("system")
        for row in history
        if isinstance(row.get("system"), str)
    }
    allowed_files = set(BASELINE_FILES) | {
        f"{name}.py" for name in prior_candidates
    }

    # Rebuild only the controlled agents directory. This prevents candidates from
    # unrelated runs (and their task-specific history) entering the blind view.
    for path in view_agents.glob("*.py"):
        if path.name not in allowed_files:
            path.unlink()
    for filename in sorted(allowed_files):
        source = AGENTS_DIR / filename
        if source.exists():
            shutil.copy2(source, view_agents / filename)

    for filename in ("__init__.py", "llm.py", "memory_system.py"):
        source = EVOLVE_DIR / filename
        if source.exists():
            shutil.copy2(source, package_dir / filename)

    results = load_results(LOGS_DIR, "val.json")
    by_system = {}
    for (_model, _dataset, system), result in results.items():
        if result.get("accuracy") is None:
            continue
        stats = by_system.setdefault(system, {"accuracies": [], "contexts": []})
        stats["accuracies"].append(result["accuracy"] * 100)
        stats["contexts"].append(result.get("memory_context_chars", 0))

    systems = []
    for system, stats in sorted(by_system.items()):
        # Only expose complete macro averages, never individual task scores.
        if len(stats["accuracies"]) != num_datasets:
            continue
        non_zero_contexts = [value for value in stats["contexts"] if value > 0]
        systems.append(
            {
                "system": system,
                "avg_val": round(sum(stats["accuracies"]) / num_datasets, 1),
                "avg_context_chars": round(
                    sum(non_zero_contexts) / len(non_zero_contexts)
                    if non_zero_contexts
                    else 0
                ),
            }
        )

    frontier = json.loads(FRONTIER_VAL.read_text(encoding="utf-8")) if FRONTIER_VAL.exists() else {}
    pareto = [
        {
            "system": row.get("system"),
            "avg_val": row.get("val_accuracy"),
            "avg_context_chars": row.get("ctx_len", 0),
        }
        for row in frontier.get("_pareto", [])
    ]
    safe_history = [
        {
            key: row[key]
            for key in (
                "iteration",
                "system",
                "avg_val",
                "axis",
                "delta",
                "outcome",
                "components",
            )
            if key in row
        }
        for row in history
    ]
    state = {
        "mode": "dataset_blind",
        "task_count": num_datasets,
        "objective": "unweighted macro-average validation accuracy",
        "systems": systems,
        "pareto": pareto,
        "history": safe_history,
    }
    (package_dir / "optimizer_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    write_anonymous_traces(package_dir)
    pending = package_dir / "pending_eval.json"
    if pending.exists():
        pending.unlink()
    return package_dir, pending


def write_anonymous_traces(package_dir):
    """Pool validation traces without exposing task names or task boundaries."""
    traces = []
    for log_path in LOGS_DIR.rglob("log.jsonl"):
        system = None
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "meta":
                system = event.get("memory")
                continue
            if event.get("type") != "val_step" or not system:
                continue
            trace = {
                "system": system,
                "input": event.get("input", ""),
                "prediction": event.get("pred"),
                "ground_truth": event.get("tgt"),
                "was_correct": bool(event.get("ok")),
                "prompt_len": event.get("prompt_len", 0),
                "context_len": event.get("context_len", 0),
                "prompt_text": event.get("prompt_text", ""),
            }
            encoded = json.dumps(trace, sort_keys=True, ensure_ascii=False)
            trace["trace_id"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
            traces.append(trace)

    # A content-derived order mixes source tasks without recording their identity or
    # retaining directory order, while remaining reproducible across resumes.
    traces.sort(key=lambda row: row["trace_id"])
    output = package_dir / "anonymous_traces.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces),
        encoding="utf-8",
    )


def install_dataset_blind_candidates(view_dir, view_pending):
    """Validate and install candidates produced inside the blind workspace."""
    if not view_pending.exists():
        print(f"  {_red('FAIL')} blind proposer did not write pending_eval.json")
        return False
    try:
        payload = json.loads(view_pending.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  {_red('FAIL')} invalid blind pending_eval.json: {exc}")
        return False

    candidates = payload.get("candidates", [])
    if not candidates:
        print(f"  {_red('FAIL')} blind proposer returned no candidates")
        return False
    for candidate in candidates:
        name = candidate.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            print(f"  {_red('FAIL')} invalid candidate name: {name!r}")
            return False
        source = view_dir / "agents" / f"{name}.py"
        target = AGENTS_DIR / f"{name}.py"
        if not source.is_file():
            print(f"  {_red('FAIL')} missing candidate source: {source}")
            return False
        if target.exists() and target.read_bytes() != source.read_bytes():
            print(f"  {_red('FAIL')} candidate would overwrite existing file: {target}")
            return False

    for candidate in candidates:
        name = candidate["name"]
        shutil.copy2(view_dir / "agents" / f"{name}.py", AGENTS_DIR / f"{name}.py")
        candidate["file"] = f"agents/{name}.py"
    PENDING_EVAL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def render_task_prompt(iteration, num_datasets, dataset_blind=False, view_dir=None):
    """Build the prompt for the proposer Claude session."""
    if dataset_blind:
        return (
            f"Run iteration {iteration} of dataset-blind memory-system evolution. "
            f"The objective combines {num_datasets} anonymous tasks.\n\n"
            "Work only inside the isolated proposer view given as your working directory. "
            "Do not inspect parent directories or any config, data, raw evaluation logs, "
            "reports, or per-task results outside it.\n\n"
            "Read `optimizer_state.json` for aggregate feedback and "
            "`anonymous_traces.jsonl` for pooled problem-level traces. Read and edit "
            "reusable systems under `agents/`. "
            "Write `pending_eval.json` in the working directory. Dataset identities and "
            "per-dataset scores are intentionally unavailable."
        )
    return (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{EVOLUTION_SUMMARY}` — past results\n"
        f"- `{FRONTIER_VAL}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )


def count_iterations_from_summary():
    """Highest iteration number in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def propose_codex(
    task_prompt,
    iteration,
    proposer,
    timeout=2400,
    dataset_blind=False,
    num_datasets=1,
):
    """Returns True if candidates were produced (pending_eval.json exists)."""
    proposer_cwd = EVOLVE_DIR
    proposer_skill = EVOLVE_DIR / ".claude/skills/meta-harness"
    view_pending = None
    if dataset_blind:
        proposer_cwd, view_pending = prepare_dataset_blind_view(num_datasets)
        proposer_skill = EVOLVE_DIR / ".claude/skills/meta-harness-blind"
        task_prompt = render_task_prompt(
            iteration,
            num_datasets,
            dataset_blind=True,
            view_dir=proposer_cwd,
        )

    result = codex_wrapper.run(
        prompt=task_prompt,
        model=proposer["model"],
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(proposer_skill)],
        cwd=str(proposer_cwd),
        log_dir=str(LOGS_DIR / "codex_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort=proposer["reasoning_effort"],
    )
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    if dataset_blind:
        return install_dataset_blind_candidates(proposer_cwd, view_pending)
    return PENDING_EVAL.exists()


def validate_candidates(candidates):
    """Import-check each candidate. Returns list of valid candidates."""
    valid = []
    for c in candidates:
        name = c["name"]
        result = run_cmd(
            [
                "uv",
                "run",
                "--project",
                str(EVOLVE_DIR),
                "python",
                "-c",
                f"from text_classification.agents.{name} import *; print('OK')",
            ],
            cwd=str(EVOLVE_DIR.parent),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    best_val,
    propose_time=None,
    bench_time=None,
    wall_time=None,
):
    """Append one JSONL row per candidate to evolution_summary.jsonl."""
    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1),
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": round(avg_val - best_val, 1) if best_val else None,
                "outcome": f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                if avg_val > 0
                else "failed",
            }
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row) + "\n")


def fresh_start():
    """Clear proposed memory systems and reset logs for a fresh run."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    if AGENTS_DIR.exists():
        files = [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]
        for f in files:
            f.unlink()
        if files:
            print(f"  Cleared {len(files)} candidate file(s) from agents/")

    for f in [
        EVOLUTION_SUMMARY,
        FRONTIER_VAL,
        LOGS_DIR / "frontier.json",
        PENDING_EVAL,
    ]:
        if f.exists():
            f.unlink()

    if LOGS_DIR.exists():
        val_files = list(LOGS_DIR.rglob("val.json"))
        for f in val_files:
            f.unlink()
        if val_files:
            print(f"  Cleared {len(val_files)} val result files")

    print(f"  {_green('Fresh start')}: cleared generated agents and log files")


def finalize_run(baselines, datasets, model_short):
    """Evaluate test once, after freezing this run against further evolution."""
    if not FRONTIER_VAL.exists():
        print(f"ERROR: no validation frontier for run at {LOGS_DIR}")
        raise SystemExit(1)

    if FINALIZED.exists():
        state = json.loads(FINALIZED.read_text())
        if state.get("status") == "complete":
            print(f"Run already finalized: {LOGS_DIR.name}")
            result = run_benchmark(["--results", "--test"])
            if result.stdout:
                print(result.stdout)
            return

    frontier = json.loads(FRONTIER_VAL.read_text())
    pareto = frontier.get("_pareto", [])
    test_systems = set(baselines)
    test_systems.update(entry["system"] for entry in pareto)
    for key, value in frontier.items():
        if not key.startswith("_") and isinstance(value, dict):
            if "best_system" in value:
                test_systems.add(value["best_system"])

    FINALIZED.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "systems": sorted(test_systems),
            },
            indent=2,
        )
    )

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")
    failed = False
    for name in sorted(test_systems):
        print(f"  {_ts()} test eval: {_bold(name)}", flush=True)
        result = run_benchmark(["--memory", name, "--test"])
        if result.returncode != 0:
            failed = True
            print(f"    {_red('FAIL')} {name} test eval failed")

    frontier_result = run_benchmark(["--frontier", "--test", "--model", model_short])
    failed = failed or frontier_result.returncode != 0

    result = run_benchmark(["--results", "--test"])
    failed = failed or result.returncode != 0
    if result.stdout:
        print(result.stdout)

    test_results = load_results(RESULTS_DIR, "test.json")
    missing = [
        (model_short, dataset, system)
        for system in test_systems
        for dataset in datasets
        if (model_short, dataset, system) not in test_results
    ]
    if missing:
        failed = True
        print(f"Missing {len(missing)} complete test result group(s).")

    if failed:
        print("Test finalization incomplete. Fix failures, then rerun --test.")
        raise SystemExit(1)

    state = json.loads(FINALIZED.read_text())
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    FINALIZED.write_text(json.dumps(state, indent=2))
    print(f"\n{_ts()} {_bold('Test finalization complete.')}")


def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global RESULTS_DIR, FINALIZED, PROPOSER_VIEW

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["datasets"]

    config_model_ids = [m["model"] for m in cfg.get("models", [])]
    if args.model not in config_model_ids:
        print(f"ERROR: --model {args.model} not in config.yaml: {config_model_ids}")
        sys.exit(1)

    model_short = get_model_short_name(args.model)

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    RESULTS_DIR = LOGS_DIR / "results"
    FINALIZED = LOGS_DIR / "finalized.json"
    PROPOSER_VIEW = LOGS_DIR / "proposer_view"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = cfg["memory_systems"]["baselines"]
    dataset_blind = bool(cfg.get("proposer", {}).get("dataset_blind", False))
    if args.test:
        finalize_run(baselines, datasets, model_short)
        return

    if (
        FINALIZED.exists()
        and json.loads(FINALIZED.read_text()).get("status") == "complete"
    ):
        print(
            f"ERROR: run '{run_name}' is finalized; use a new --run-name to continue evolution"
        )
        raise SystemExit(1)

    if args.fresh:
        fresh_start()

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  model={_cyan(args.model)}  "
        f"iters={args.iterations}  datasets={datasets}  "
        f"optimization={'dataset-blind' if dataset_blind else 'dataset-aware'}"
    )

    # ── Phase 0: Baselines ─────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl])
            elapsed = time.time() - t0
            if result.returncode != 0:
                failure_log = LOGS_DIR / f"baseline_{bl}_failure.log"
                failure_log.write_text(
                    (result.stdout or "") + "\n" + (result.stderr or ""),
                    encoding="utf-8",
                )
                print(f"    {_red('FAIL')} {bl}: {result.stderr[-2000:]}")
                print(f"Full error: {failure_log}")
                raise SystemExit(1)
            else:
                print(f"    {_green('OK')} ({_elapsed(elapsed)})")

        run_benchmark(["--frontier", "--model", model_short])

        # Show baseline results
        results = load_results(LOGS_DIR, "val.json")
        for bl in baselines:
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, bl)]
                if k in results and results[k].get("accuracy") is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {_bold(bl)}: avg_val={_pct(avg)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        # Show frontier status
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        task_prompt = render_task_prompt(iteration, len(datasets))

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        # Propose
        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        ok = propose_codex(
            task_prompt,
            iteration,
            cfg["proposer"],
            timeout=args.propose_timeout,
            dataset_blind=dataset_blind,
            num_datasets=len(datasets),
        )
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        valid_candidates = validate_candidates(candidates)

        if not valid_candidates:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration,
                candidates,
                {},
                best_val,
                propose_time=propose_time,
            )
            continue
        print(
            f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
        )

        # Benchmark
        bench_start = time.time()
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(datasets)} datasets"
        )
        for ci, c in enumerate(valid_candidates):
            if _interrupted:
                break
            name = c["name"]
            print(
                f"    [{ci + 1}/{len(valid_candidates)}] {_bold(name)}...", flush=True
            )
            t0 = time.time()
            result = run_benchmark(["--memory", name])
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"      {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})")
            else:
                print(f"      {_green('OK')} ({_elapsed(elapsed)})")
        bench_time = time.time() - bench_start

        run_benchmark(["--frontier", "--model", model_short])

        # Compute scores and show results
        val_scores = {}
        results = load_results(LOGS_DIR, "val.json")
        for c in valid_candidates:
            name = c["name"]
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, name)]
                if k in results and results[k].get("accuracy") is not None
            ]
            val_scores[name] = sum(accs) / len(accs) if accs else 0
            delta = val_scores[name] - (best_val * 100 if best_val <= 1 else best_val)
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}"
            )

        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            best_val,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
        )

        # Show iteration summary
        improved = any(
            v > (best_val * 100 if best_val <= 1 else best_val)
            for v in val_scores.values()
        )
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Validation evolution complete.')}")
    print(
        f"Finalize once with: uv run python meta_harness.py --run-name {run_name} --test"
    )


def main():
    parser = argparse.ArgumentParser(description="Evolution loop for memory systems")
    parser.add_argument("--iterations", type=int, default=20)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _default_model = _cfg["models"][0]["model"] if _cfg.get("models") else None
    parser.add_argument(
        "--model",
        default=_default_model,
        help=f"Solver model (default: {_default_model})",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout per propose step (default: 2400s)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs. Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed systems and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Finalize an existing named run with one held-out test evaluation",
    )
    args = parser.parse_args()

    if args.test and not args.run_name:
        parser.error("--test requires --run-name")
    if args.test and args.fresh:
        parser.error("--test cannot be combined with --fresh")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
