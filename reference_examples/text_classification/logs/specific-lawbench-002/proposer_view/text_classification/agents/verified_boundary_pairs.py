"""Retrieve pairs of similar cases whose verified answers differ."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The labeled cases are selected because they resemble the current problem. Some similar cases have different verified answers, so compare the facts that distinguish them. The answers below are ground truth; previous model guesses are not included. Follow the current problem's own instructions.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _features(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.update(run[i : i + 2] for i in range(len(run) - 1))
            result.update(run[i : i + 3] for i in range(len(run) - 2))
        else:
            result[run] += 1
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


def _clip(text: str, limit: int = 900) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


class Candidate(MemorySystem):
    """Build query-relevant, ground-truth-only boundary pairs at prediction time."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 7600
        self.max_pairs = 3
        self._last_context = ""

    def _vectors(self) -> tuple[list[dict[str, float]], dict[str, float]]:
        documents = [_features(example["input"]) for example in self.examples]
        frequency: Counter[str] = Counter()
        for document in documents:
            frequency.update(document.keys())
        count = len(documents)
        inverse = {
            token: math.log(1.0 + (count + 1) / (value + 1))
            for token, value in frequency.items()
        }
        vectors: list[dict[str, float]] = []
        for document in documents:
            total = sum(document.values()) or 1
            raw = {token: value / total * inverse[token] for token, value in document.items()}
            norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
            vectors.append({token: value / norm for token, value in raw.items()})
        return vectors, inverse

    def _select_pairs(self, query: str) -> list[tuple[int, int]]:
        if len(self.examples) < 2:
            return []
        vectors, inverse = self._vectors()
        query_counts = _features(query)
        query_raw = {token: value * inverse.get(token, 0.0) for token, value in query_counts.items()}
        norm = math.sqrt(sum(value * value for value in query_raw.values())) or 1.0
        query_vector = {token: value / norm for token, value in query_raw.items()}
        query_scores = [_cosine(query_vector, vector) for vector in vectors]
        pairs: list[tuple[float, int, int]] = []
        for left in range(len(self.examples)):
            for right in range(left + 1, len(self.examples)):
                if self.examples[left]["target"] == self.examples[right]["target"]:
                    continue
                both = min(query_scores[left], query_scores[right])
                if both <= 0:
                    continue
                pair_similarity = _cosine(vectors[left], vectors[right])
                score = 0.65 * both + 0.35 * (query_scores[left] + query_scores[right]) / 2
                score += 0.20 * pair_similarity
                pairs.append((score, left, right))
        pairs.sort(reverse=True)
        selected: list[tuple[int, int]] = []
        used: set[int] = set()
        for _, left, right in pairs:
            if left in used or right in used:
                continue
            selected.append((left, right))
            used.update((left, right))
            if len(selected) >= self.max_pairs:
                break
        if not selected:
            nearest = sorted(range(len(self.examples)), key=lambda i: (query_scores[i], i), reverse=True)
            if len(nearest) >= 2:
                selected.append((nearest[0], nearest[1]))
        return selected

    def _format_examples(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        for pair_number, (left, right) in enumerate(self._select_pairs(query), start=1):
            first, second = self.examples[left], self.examples[right]
            part = (
                f"Similar case pair {pair_number}:\n"
                f"Case A: {_clip(first['input'])}\nVerified answer A: {first['target']}\n"
                f"Case B: {_clip(second['input'])}\nVerified answer B: {second['target']}"
            )
            remaining = self.context_budget - used
            if remaining < 150:
                break
            if len(part) > remaining:
                part = part[: remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No paired examples are available yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_pairs": self._last_context.count("Similar case pair"),
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(example["input"], example["target"]) for example in self.examples}
        for row in batch_results:
            item = {"input": _source_text(row), "target": str(row["ground_truth"])}
            key = (item["input"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._last_context = ""
