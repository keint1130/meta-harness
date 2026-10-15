"""Evaluate one candidate math harness on one versioned split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from math_harness import SolvedMathMemory, instantiate_candidate
from splits import ROOT, load_config, select_rows


def math_repo_path(config: dict) -> Path:
    return (ROOT / config["paths"]["math_eval_repo"]).resolve()


def install_math_runtime(config: dict):
    repo = math_repo_path(config)
    if not repo.exists():
        raise FileNotFoundError(f"math-frontier-eval not found: {repo}")
    sys.path.insert(0, str(repo))
    # Importing benchmark installs the GPT-6 Chat Completions compatibility shim.
    import benchmark as math_benchmark  # noqa: F401
    from grading import verify
    from llm_provider import LLM

    return LLM, verify


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(f"math_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate_class = getattr(module, "Candidate", None)
    if candidate_class is None:
        raise TypeError(f"{path} must expose class Candidate")
    return candidate_class


def _response_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _response_text(value[0])
    return "" if value is None else str(value)


def iter_batches(items: list[str], batch_size: int):
    if batch_size < 1:
        raise ValueError("progress_batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def partial_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".partial.json")


def load_partial(output_path: Path, expected: dict) -> dict:
    path = partial_path(output_path)
    if not path.exists():
        return {
            **expected,
            "responses_by_problem_id": {},
            "input_tokens": 0,
            "output_tokens": 0,
        }
    partial = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if partial.get(key) != value:
            raise ValueError(f"Partial result mismatch for {key}: {path}")
    return partial


def partial_responses_by_id(partial: dict, problem_ids: list[str]) -> dict:
    """Read current sparse checkpoints and the older prefix-list format."""
    if "responses_by_problem_id" in partial:
        return dict(partial["responses_by_problem_id"])
    responses = partial.get("responses", [])
    return {
        problem_id: response
        for problem_id, response in zip(problem_ids, responses)
    }


def load_reused_responses(
    source: Path,
    problem_ids: list[str],
    regenerate_ids: set[str],
) -> tuple[dict, int, int]:
    result = json.loads(source.read_text(encoding="utf-8"))
    details = result.get("details", [])
    by_id = {item["problem_id"]: item["response"] for item in details}
    missing = set(problem_ids).difference(by_id)
    if missing:
        raise ValueError(f"Reuse result is missing problem IDs: {sorted(missing)}")
    unknown = regenerate_ids.difference(problem_ids)
    if unknown:
        raise ValueError(f"Unknown regenerate IDs: {sorted(unknown)}")
    reused = {
        problem_id: by_id[problem_id]
        for problem_id in problem_ids
        if problem_id not in regenerate_ids
    }
    return (
        reused,
        int(result.get("solver_input_tokens", 0)),
        int(result.get("solver_output_tokens", 0)),
    )


def save_partial(output_path: Path, partial: dict) -> None:
    path = partial_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partial, indent=2), encoding="utf-8")


def judge_prompt(problem: str, groundtruth: str, response: str) -> str:
    return f"""Decide whether the student's final mathematical answer is equivalent to the reference answer.

Allow equivalent notation, reordered finite sets, renamed arbitrary variables, labelled equalities, reparameterized complete solution families, and logically equivalent descriptions of a solution set.

Require the student's claimed final answer to be complete and equivalent to the reference answer. A recurrence, algorithm, unevaluated optimization, bound, or proof strategy is incorrect when the problem asks for an explicit value or closed form. A finite list is incorrect when the reference describes a larger infinite family. Do not accept an answer merely because intermediate work matches or because some listed cases are correct.

Problem:
{problem}

Reference answer:
{groundtruth}

Student response:
{response}

Return exactly one of these and nothing else:
\\boxed{{correct}}
\\boxed{{incorrect}}"""


def parse_judge_response(response: str) -> float:
    value = response.lower().replace(" ", "")
    if "\\boxed{incorrect}" in value:
        return 0.0
    if "\\boxed{correct}" in value:
        return 1.0
    return 0.0


def evaluate(
    candidate_path: Path,
    split_name: str,
    domain: str | None,
    output_path: Path,
    reuse_result: Path | None = None,
    regenerate_ids: set[str] | None = None,
    problem_limit: int | None = None,
) -> dict:
    config = load_config()
    rows = select_rows(split_name, domain, config)
    if problem_limit is not None:
        if problem_limit < 1:
            raise ValueError("problem_limit must be at least 1")
        rows = rows[:problem_limit]
    LLM, verify = install_math_runtime(config)
    memory_cfg = config.get("memory", {})
    memory = SolvedMathMemory(
        memory_cfg.get("repository", "yoonholee/math-corpus-combined"),
        memory_cfg.get("split", "train"),
        memory_cfg.get("revision"),
    )
    print(
        f"Preparing {candidate_path.stem}: candidate construction and retrieval index "
        "(the external memory corpus is loaded only if used)",
        flush=True,
    )
    candidate = instantiate_candidate(
        load_candidate(candidate_path), rows, domain=domain, memory=memory
    )
    prompts = [candidate.build_prompt(row["problem"]) for row in rows]
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("Every build_prompt result must be a non-empty string")
    print(f"  Prepared {len(prompts)} prompts", flush=True)

    experiment = config["experiment"]
    started = time.monotonic()
    generation_kwargs = {
        "max_tokens": experiment["max_tokens"],
        "cache": experiment["cache"],
        "reasoning_effort": experiment["reasoning_effort"],
    }
    if experiment.get("temperature") is not None:
        generation_kwargs["temperature"] = experiment["temperature"]
    batch_size = int(experiment.get("progress_batch_size", experiment["concurrency"]))
    print(
        f"Starting {candidate_path.stem}: split={split_name} "
        f"domain={domain or 'global'} problems={len(rows)} "
        f"model={experiment['model']} reasoning={experiment['reasoning_effort']} "
        f"concurrency={experiment['concurrency']}",
        flush=True,
    )
    checkpoint_identity = {
        "candidate": candidate_path.stem,
        "split": split_name,
        "domain": domain or "global",
        "model": experiment["model"],
        "reasoning_effort": experiment["reasoning_effort"],
        "problem_ids": [row["problem_id"] for row in rows],
    }
    partial = load_partial(output_path, checkpoint_identity)
    problem_ids = checkpoint_identity["problem_ids"]
    response_by_id = partial_responses_by_id(partial, problem_ids)
    regenerate_ids = regenerate_ids or set()
    migration_input_tokens = int(
        partial.get("migration_input_tokens", partial.get("input_tokens", 0))
    )
    migration_output_tokens = int(
        partial.get("migration_output_tokens", partial.get("output_tokens", 0))
    )
    if reuse_result:
        reused, reused_input_tokens, reused_output_tokens = load_reused_responses(
            reuse_result, problem_ids, regenerate_ids
        )
        # A newly generated checkpoint response always wins over an older result.
        response_by_id = {**reused, **response_by_id}
        input_tokens = reused_input_tokens
        output_tokens = reused_output_tokens
        print(
            f"  Reusing {len(reused)} responses from {reuse_result}",
            flush=True,
        )
    else:
        input_tokens = int(partial.get("input_tokens", 0))
        output_tokens = int(partial.get("output_tokens", 0))
    if response_by_id:
        print(
            f"  Starting with {len(response_by_id)}/{len(rows)} completed responses",
            flush=True,
        )

    remaining = [
        (row["problem_id"], prompt)
        for row, prompt in zip(rows, prompts)
        if row["problem_id"] not in response_by_id
    ]
    for item_batch in iter_batches(remaining, batch_size):
        batch_ids = [item[0] for item in item_batch]
        prompt_batch = [item[1] for item in item_batch]
        # llm-provider's async OpenAI client cannot safely survive repeated
        # asyncio.run() calls. A fresh context per progress batch also makes a
        # transient transport failure resumable at a clean boundary.
        with LLM(model=experiment["model"], max_concurrent=experiment["concurrency"]) as llm:
            batch_results = llm.generate(prompt_batch, **generation_kwargs)
            input_tokens += llm.total_input_tokens
            output_tokens += llm.total_output_tokens
            migration_input_tokens += llm.total_input_tokens
            migration_output_tokens += llm.total_output_tokens
        response_by_id.update(zip(batch_ids, batch_results))
        partial = {
            **checkpoint_identity,
            "responses_by_problem_id": response_by_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "migration_input_tokens": migration_input_tokens,
            "migration_output_tokens": migration_output_tokens,
        }
        save_partial(output_path, partial)
        completed = len(response_by_id)
        empty_responses = sum(
            not _response_text(value).strip() for value in response_by_id.values()
        )
        batch_elapsed = time.monotonic() - started
        print(
            f"  [{completed:>3}/{len(rows)}] "
            f"elapsed={batch_elapsed:.1f}s "
            f"tokens={input_tokens}in/{output_tokens}out "
            f"empty={empty_responses}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    responses = [_response_text(response_by_id[problem_id]) for problem_id in problem_ids]
    if len(responses) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} responses, got {len(responses)}")

    details = []
    for row, prompt, response in zip(rows, prompts, responses):
        score = float(verify(response, row["groundtruth"]))
        details.append(
            {
                "problem_id": row["problem_id"],
                "domain": row["domain"],
                "subcategory": row["subcategory"],
                "groundtruth": row["groundtruth"],
                "problem": row["problem"],
                "prompt": prompt,
                "response": response,
                "deterministic_score": score,
                "judge_score": None,
                "judge_response": None,
                "score": score,
            }
        )

    judge_cfg = config.get("judge", {})
    judge_targets = [
        item
        for item in details
        if judge_cfg.get("enabled", False)
        and item["deterministic_score"] == 0.0
        and item["response"].strip()
    ]
    judge_input_tokens = 0
    judge_output_tokens = 0
    if judge_targets:
        print(
            f"Judging {len(judge_targets)} non-empty deterministic failures with "
            f"{judge_cfg['model']} reasoning={judge_cfg['reasoning_effort']}",
            flush=True,
        )
        judge_kwargs = {
            "max_tokens": judge_cfg["max_tokens"],
            "reasoning_effort": judge_cfg["reasoning_effort"],
            "cache": judge_cfg.get("cache", False),
        }
        judge_batch_size = int(judge_cfg.get("batch_size", judge_cfg["concurrency"]))
        for start in range(0, len(judge_targets), judge_batch_size):
            target_batch = judge_targets[start : start + judge_batch_size]
            prompt_batch = [
                judge_prompt(item["problem"], item["groundtruth"], item["response"])
                for item in target_batch
            ]
            with LLM(model=judge_cfg["model"], max_concurrent=judge_cfg["concurrency"]) as judge_llm:
                judge_raw = judge_llm.generate(prompt_batch, **judge_kwargs)
                judge_input_tokens += judge_llm.total_input_tokens
                judge_output_tokens += judge_llm.total_output_tokens
            for item, raw_value in zip(target_batch, judge_raw):
                judge_text = _response_text(raw_value)
                item["judge_response"] = judge_text
                item["judge_score"] = parse_judge_response(judge_text)
                item["score"] = max(item["deterministic_score"], item["judge_score"])
            print(
                f"  judge [{min(start + len(target_batch), len(judge_targets)):>3}/"
                f"{len(judge_targets)}] tokens={judge_input_tokens}in/"
                f"{judge_output_tokens}out",
                flush=True,
            )

    domain_scores = {}
    for selected_domain in config["domains"]:
        scores = [item["score"] for item in details if item["domain"] == selected_domain]
        if scores:
            domain_scores[selected_domain] = sum(scores) / len(scores)
    accuracy = sum(item["score"] for item in details) / len(details)
    macro_accuracy = sum(domain_scores.values()) / len(domain_scores)
    result = {
        "candidate": candidate_path.stem,
        "candidate_path": str(candidate_path.resolve()),
        "split": split_name,
        "domain": domain or "global",
        "model": experiment["model"],
        "n_problems": len(rows),
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "domain_accuracy": domain_scores,
        "solver_input_tokens": input_tokens,
        "solver_output_tokens": output_tokens,
        "migration_input_tokens": migration_input_tokens if reuse_result else None,
        "migration_output_tokens": migration_output_tokens if reuse_result else None,
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
        "total_input_tokens": input_tokens + judge_input_tokens,
        "total_output_tokens": output_tokens + judge_output_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "empty_responses": sum(not response.strip() for response in responses),
        "reused_result": str(reuse_result.resolve()) if reuse_result else None,
        "regenerated_problem_ids": sorted(regenerate_ids),
        "candidate_source_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "evaluation_config": {
            "experiment": config["experiment"],
            "judge": config.get("judge", {}),
            "memory": config.get("memory", {}),
            "splits": config["splits"],
        },
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint = partial_path(output_path)
    if checkpoint.exists():
        checkpoint.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--split", choices=("search", "dev", "test"), required=True)
    parser.add_argument("--domain", choices=load_config()["domains"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-result", type=Path)
    parser.add_argument("--regenerate-id", action="append", default=[])
    parser.add_argument("--problem-limit", type=int)
    args = parser.parse_args()
    result = evaluate(
        args.candidate,
        args.split,
        args.domain,
        args.output,
        reuse_result=args.reuse_result,
        regenerate_ids=set(args.regenerate_id),
        problem_limit=args.problem_limit,
    )
    print(
        f"{result['candidate']} {result['domain']} {result['split']}: "
        f"accuracy={result['accuracy'] * 100:.1f}% "
        f"macro={result['macro_accuracy'] * 100:.1f}% "
        f"n={result['n_problems']}"
    )


if __name__ == "__main__":
    main()

