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
    with LLM(model=experiment["model"], max_concurrent=experiment["concurrency"]) as llm:
        raw = llm.generate(prompts, **generation_kwargs)
        input_tokens = llm.total_input_tokens
        output_tokens = llm.total_output_tokens
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

