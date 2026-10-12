"""Similarity-gated replay of prior errors as corrected contrastive examples."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Learn from the feedback examples below. In every correction, the model answer is known to be wrong and the corrected answer is authoritative. Use a similar correction to avoid repeating its mistake. The examples are guidance; follow the current problem's instructions.

{memory}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]"}}"""


def _features(text: str) -> set[str]:
    result: set[str] = set()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.update(run[i : i + 2] for i in range(len(run) - 1))
            result.update(run[i : i + 3] for i in range(len(run) - 2))
        else:
            result.add(run)
    return result


def _source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        row.get("raw_question")
        or metadata.get("raw_question")
        or metadata.get("question")
        or row.get("input", "")
    )


def _clip(text: str, limit: int = 1050) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


class Candidate(MemorySystem):
    """Store feedback outcomes and retrieve the most relevant corrections."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.context_budget = 9000
        self._last_context_len = 0

    def _rank(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        query_features = _features(query)
        if not query_features:
            return rows[-5:][::-1]
        row_features = [_features(row["input"]) for row in rows]
        df: Counter[str] = Counter()
        for features in row_features:
            df.update(features)
        n_rows = len(rows)
        weights = {
            token: math.log(1.0 + (n_rows - df[token] + 0.5) / (df[token] + 0.5))
            for token in query_features
        }
        denom = sum(weights.values()) or 1.0
        scores = [
            (sum(weights[token] for token in query_features & features) / denom, i)
            for i, features in enumerate(row_features)
        ]
        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [rows[i] for score, i in scores[:6] if score > 0]

    def _format_memory(self, query: str) -> str:
        failures = [row for row in self.examples if not row["was_correct"]]
        successes = [row for row in self.examples if row["was_correct"]]
        corrections = self._rank(query, failures)
        anchors = self._rank(query, successes)[:2]
        parts: list[str] = []
        used = 0
        for row in corrections:
            bad = _clip(str(row.get("prediction", "")), 180)
            part = (
                f"Similar case: {_clip(row['input'])}\n"
                f"Known wrong model answer: {bad}\n"
                f"Corrected answer: {row['target']}"
            )
            if used + len(part) > self.context_budget:
                continue
            parts.append(part)
            used += len(part) + 2
        for row in anchors:
            part = f"Related labeled case: {_clip(row['input'], 700)}\nAnswer: {row['target']}"
            if used + len(part) > self.context_budget:
                continue
            parts.append(part)
            used += len(part) + 2
        self._last_context_len = used
        return "\n\n".join(parts) or "No feedback examples yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        memory = self._format_memory(input)
        response = self.call_llm(_PROMPT.format(memory=memory, input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "correction_examples": sum(not row["was_correct"] for row in self.examples),
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            self.examples.append(
                {
                    "input": _source_text(row),
                    "prediction": str(row.get("prediction", "")),
                    "target": str(row["ground_truth"]),
                    "was_correct": bool(row.get("was_correct", False)),
                }
            )

    def get_context_length(self) -> int:
        return self._last_context_len

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])

