"""Use a draft answer and retrieved feedback in a separate verification pass."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_DRAFT_PROMPT = """Use the related labeled examples as guidance. Their facts may differ, so apply the current problem's instructions.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""
_REVIEW_PROMPT = """Review the proposed answer against the current problem and the verified feedback below. Change it only when the evidence supports a correction. Check that the answer is complete and follows the requested answer form; preserve the exact wording of a verified output when it applies. Similar examples can differ in decisive facts.

Verified feedback:
{feedback}

Current problem:
{input}

Proposed answer:
{draft}

Return JSON with the reviewed concise final_answer.
{{"reasoning": "[brief verification]", "final_answer": "[answer]}}"""


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
    """Retrieve both labeled cases and prior errors, then audit an LLM draft."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.context_budget = 6500
        self.max_examples = 6
        self._last_context = ""
        self._last_feedback = ""

    def _retrieve(self, query: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        query_features = _features(query)
        row_features = [_features(row["input"]) for row in rows]
        df: Counter[str] = Counter()
        for features in row_features:
            df.update(features)
        n_rows = len(rows)
        weights = {
            token: math.log(1.0 + (n_rows - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in df.items()
        }
        denom = sum(weights.get(token, 0.0) for token in query_features) or 1.0
        scored = []
        for index, features in enumerate(row_features):
            score = sum(weights.get(token, 0.0) for token in query_features & features) / denom
            scored.append((score, index))
        scored.sort(reverse=True)
        return [rows[index] for score, index in scored[:limit] if score > 0]

    def _format_examples(self, query: str) -> str:
        selected = self._retrieve(query, self.examples, self.max_examples)
        parts: list[str] = []
        used = 0
        for row in selected:
            part = f"Related problem: {_clip(row['input'])}\nVerified answer: {row['target']}"
            if used + len(part) + 2 > self.context_budget:
                continue
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No learned examples yet."

    def _format_corrections(self, query: str) -> str:
        failures = [row for row in self.examples if not row["was_correct"]]
        selected = self._retrieve(query, failures, 4)
        parts: list[str] = []
        for row in selected:
            parts.append(
                f"Related prior error:\nProblem: {_clip(row['input'], 700)}\n"
                f"Previous incorrect answer: {_clip(row['prediction'], 180)}\n"
                f"Verified correction: {row['target']}"
            )
        self._last_feedback = "\n\n".join(parts)
        return self._last_feedback or "No similar prior errors are available."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        draft_response = self.call_llm(_DRAFT_PROMPT.format(examples=examples, input=input))
        draft = extract_json_field(draft_response, "final_answer")
        feedback = self._format_corrections(input)
        review_response = self.call_llm(
            _REVIEW_PROMPT.format(feedback=feedback, input=input, draft=draft)
        )
        answer = extract_json_field(review_response, "final_answer", default=draft)
        if not answer:
            answer = draft
        return answer, {
            "full_response": review_response,
            "draft_response": draft_response,
            "reviewed": True,
            "retrieved_examples": len(self._last_context.split("\n\n")) if self._last_context else 0,
            "prior_errors": sum(not row["was_correct"] for row in self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            self.examples.append(
                {
                    "input": _source_text(row),
                    "target": str(row["ground_truth"]),
                    "prediction": str(row.get("prediction", "")),
                    "was_correct": bool(row.get("was_correct", False)),
                }
            )

    def get_context_length(self) -> int:
        return len(self._last_context) + len(self._last_feedback)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
        self._last_feedback = ""
