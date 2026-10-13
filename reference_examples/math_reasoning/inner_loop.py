"""Evaluate one candidate math harness on one versioned split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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
        return {**expected, "responses": [], "input_tokens": 0, "output_tokens": 0}
    partial = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if partial.get(key) != value:
            raise ValueError(f"Partial result mismatch for {key}: {path}")
    return partial


def save_partial(output_path: Path, partial: dict) -> None:
    path = partial_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partial, indent=2), encoding="utf-8")


def evaluate(
    candidate_path: Path,
    split_name: str,
    domain: str | None,
    output_path: Path,
) -> dict:
    config = load_config()
    rows = select_rows(split_name, domain, config)
    LLM, verify = install_math_runtime(config)
    candidate = load_candidate(candidate_path)(test_problems=rows)
    prompts = [candidate.build_prompt(row["problem"]) for row in rows]
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("Every build_prompt result must be a non-empty string")

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
    raw = partial["responses"]
    input_tokens = int(partial.get("input_tokens", 0))
    output_tokens = int(partial.get("output_tokens", 0))
    if raw:
        print(f"  Resuming from checkpoint: {len(raw)}/{len(rows)} complete", flush=True)

    remaining_prompts = prompts[len(raw) :]
    for prompt_batch in iter_batches(remaining_prompts, batch_size):
        # llm-provider's async OpenAI client cannot safely survive repeated
        # asyncio.run() calls. A fresh context per progress batch also makes a
        # transient transport failure resumable at a clean boundary.
        with LLM(model=experiment["model"], max_concurrent=experiment["concurrency"]) as llm:
            batch_results = llm.generate(prompt_batch, **generation_kwargs)
            input_tokens += llm.total_input_tokens
            output_tokens += llm.total_output_tokens
        raw.extend(batch_results)
        partial = {
            **checkpoint_identity,
            "responses": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        save_partial(output_path, partial)
        completed = len(raw)
        batch_elapsed = time.monotonic() - started
        print(
            f"  [{completed:>3}/{len(rows)}] "
            f"elapsed={batch_elapsed:.1f}s "
            f"tokens={input_tokens}in/{output_tokens}out",
            flush=True,
        )
    elapsed = time.monotonic() - started
    responses = [_response_text(value) for value in raw]
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
                "prompt": prompt,
                "response": response,
                "score": score,
            }
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
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "elapsed_seconds": round(elapsed, 1),
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
    args = parser.parse_args()
    result = evaluate(args.candidate, args.split, args.domain, args.output)
    print(
        f"{result['candidate']} {result['domain']} {result['split']}: "
        f"accuracy={result['accuracy'] * 100:.1f}% "
        f"macro={result['macro_accuracy'] * 100:.1f}% "
        f"n={result['n_problems']}"
    )


if __name__ == "__main__":
    main()

