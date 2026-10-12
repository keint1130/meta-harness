"""Retrieve hard, cross-label example pairs near the current problem."""

import json
import math
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem by checking it against these boundary pairs.

{pairs_section}

**Problem:**
{input}

Each pair contains similar earlier problems with different correct outputs.
Identify the detail that separates the two cases, then compare the complete
problem before choosing an answer. Respond in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

MAX_BOUNDARY_PAIRS = 24
MAX_CONTEXT_PAIRS = 4


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{2,}", text.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


class BoundaryPairMemory(MemorySystem):
    """Build a compact bank of similar examples carrying different labels."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.boundary_pairs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _pair_key(left: str, right: str) -> str:
        return json.dumps(sorted((left, right)), ensure_ascii=False)

    def _add_example(self, example: dict[str, str]) -> None:
        current_tokens = _tokens(example["input"])
        for prior in self.examples:
            if prior["target"] == example["target"]:
                continue
            similarity = _overlap(current_tokens, _tokens(prior["input"]))
            key = self._pair_key(prior["target"], example["target"])
            existing = self.boundary_pairs.get(key)
            if existing is None or similarity > existing["similarity"]:
                self.boundary_pairs[key] = {
                    "left": prior,
                    "right": example,
                    "similarity": similarity,
                }
        self.examples.append(example)

    def _idf(self) -> dict[str, float]:
        document_frequency: dict[str, int] = {}
        for example in self.examples:
            for token in _tokens(example["input"]):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = max(1, len(self.examples))
        return {
            token: math.log((total + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    @staticmethod
    def _vector(text: str, idf: dict[str, float]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for token in re.findall(r"[a-z]{2,}", text.lower()):
            counts[token] = counts.get(token, 0) + 1
        vector = {token: count * idf.get(token, 1.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {token: value / norm for token, value in vector.items()} if norm else {}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def _select_pairs(self, query: str) -> list[dict[str, Any]]:
        if not self.boundary_pairs:
            return []

        idf = self._idf()
        query_vector = self._vector(query, idf)
        candidates = sorted(
            self.boundary_pairs.values(),
            key=lambda pair: pair["similarity"],
            reverse=True,
        )[:MAX_BOUNDARY_PAIRS]
        if not query:
            return candidates[:MAX_CONTEXT_PAIRS]
        ranked = []
        for pair in candidates:
            left, right = pair["left"], pair["right"]
            left_score = self._cosine(query_vector, self._vector(left["input"], idf))
            right_score = self._cosine(query_vector, self._vector(right["input"], idf))
            ranked.append((max(left_score, right_score), pair))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [pair for score, pair in ranked[:MAX_CONTEXT_PAIRS] if score > 0]

    def _format_pairs(self, query: str) -> str:
        selected = self._select_pairs(query)
        if not selected:
            return "No learned boundary pairs are available for this problem."

        lines = ["Similar problems with contrasting labels:"]
        for index, pair in enumerate(selected, start=1):
            left, right = pair["left"], pair["right"]
            lines.append(
                f"Pair {index}:\n"
                f"- Problem: {left['input']}\n  Correct output: {left['target']}\n"
                f"- Problem: {right['input']}\n  Correct output: {right['target']}"
            )
        return "\n\n".join(lines)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        pairs_section = self._format_pairs(input)
        response = self.call_llm(PROMPT_TEMPLATE.format(pairs_section=pairs_section, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_boundary_pairs": len(self.boundary_pairs),
            "num_retrieved_pairs": len(self._select_pairs(input)),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self._add_example(
                {
                    "input": str(result.get("raw_question") or result["input"]),
                    "target": str(result["ground_truth"]),
                }
            )
        if len(self.boundary_pairs) > MAX_BOUNDARY_PAIRS:
            kept = sorted(
                self.boundary_pairs.items(),
                key=lambda item: item[1]["similarity"],
                reverse=True,
            )[:MAX_BOUNDARY_PAIRS]
            self.boundary_pairs = dict(kept)

    def get_context_length(self) -> int:
        return len(self._format_pairs(""))

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "boundary_pairs": list(self.boundary_pairs.values()),
            },
            ensure_ascii=False,
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        pairs = data.get("boundary_pairs")
        if pairs is None:
            self.boundary_pairs = {}
            examples = self.examples
            self.examples = []
            for example in examples:
                self._add_example(example)
        else:
            self.boundary_pairs = {
                self._pair_key(pair["left"]["target"], pair["right"]["target"]): pair
                for pair in pairs
            }
