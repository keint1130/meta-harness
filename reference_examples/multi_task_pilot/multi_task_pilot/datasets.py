from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import EvalCase, Task, TaskFamily


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _coding_case(row: dict[str, Any]) -> EvalCase:
    prompt = row["question_content"]
    starter_code = row.get("starter_code", "").strip()
    if starter_code:
        prompt += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    prompt += (
        "\n\nWrite a complete Python 3 solution. Return only the final code in a "
        "Python code block."
    )
    return EvalCase(
        task=Task(
            id=str(row["question_id"]),
            family=TaskFamily.CODING,
            prompt=prompt,
            metadata={
                "title": row.get("question_title", ""),
                "difficulty": row.get("difficulty", ""),
                "public_test_cases": _decode_json(row.get("public_test_cases", "[]")),
            },
        ),
        reference={
            # This is base64+zlib+pickle upstream. Never unpickle it on the host;
            # the coding evaluator decodes it inside the isolated container.
            "private_test_cases_encoded": row.get("private_test_cases", ""),
            "metadata": _decode_json(row.get("metadata", "{}")),
        },
    )


def _retrieval_case(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        task=Task(
            id=str(row["question_id"]),
            family=TaskFamily.RETRIEVAL,
            prompt=(
                f"Answer the following question with a short entity or phrase.\n\n"
                f"Question: {row['question']}"
            ),
        ),
        reference=list(row["ground_truth"]),
        resources={"documents": list(row.get("context", []))},
    )


def _math_case(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        task=Task(
            id=str(row["unique_id"]),
            family=TaskFamily.MATH,
            prompt=(
                f"Solve the problem. Show your reasoning, then put the final answer in "
                f"\\boxed{{}}.\n\n{row['problem']}"
            ),
            metadata={"subject": row["subject"], "level": row["level"]},
        ),
        reference=row["answer"],
    )


LOADERS = {
    TaskFamily.CODING: _coding_case,
    TaskFamily.RETRIEVAL: _retrieval_case,
    TaskFamily.MATH: _math_case,
}


def load_cases(prepared_root: Path, family: TaskFamily, split: str) -> list[EvalCase]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    dataset_name = {
        TaskFamily.CODING: "livecodebench",
        TaskFamily.RETRIEVAL: "popqa",
        TaskFamily.MATH: "math500",
    }[family]
    path = prepared_root / dataset_name / f"{split}.jsonl"
    return [LOADERS[family](row) for row in _read_jsonl(path)]
