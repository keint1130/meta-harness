"""Retrieve a small, label-diverse set of demonstrations for each input."""

import hashlib
import json
import math
import random
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MAX_EXAMPLES = 5
MAX_PER_LABEL = 2
MAX_EXAMPLE_CHARS = 3200


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class RetrievalDiverse(MemorySystem):
    """Select relevant examples while preventing one label from dominating."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_length = 0

    def _ranked_examples(self, input: str) -> list[tuple[float, int, dict[str, str]]]:
        query = _tokens(input)
        if not self.examples:
            return []

        docs = [_tokens(ex.get("raw_question", ex["input"])) for ex in self.examples]
        df: dict[str, int] = {}
        for terms in docs:
            for term in terms:
                df[term] = df.get(term, 0) + 1

        def weight(term: str) -> float:
            return 1.0 + math.log((len(docs) + 1.0) / (df.get(term, 0) + 1.0))

        query_norm = math.sqrt(sum(weight(term) ** 2 for term in query)) or 1.0
        ranked = []
        for index, (example, terms) in enumerate(zip(self.examples, docs)):
            common = query & terms
            doc_norm = math.sqrt(sum(weight(term) ** 2 for term in terms)) or 1.0
            score = sum(weight(term) ** 2 for term in common) / (query_norm * doc_norm)
            ranked.append((score, index, example))
        # Newer items break ties, while meaningful lexical similarity dominates.
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def _select(self, input: str) -> list[tuple[float, dict[str, str]]]:
        ranked = self._ranked_examples(input)
        selected: list[tuple[float, dict[str, str]]] = []
        label_counts: dict[str, int] = {}
        used: set[int] = set()

        # First pass caps repeated labels; a second pass fills any remaining slots.
        for enforce_diversity in (True, False):
            for score, index, example in ranked:
                if index in used:
                    continue
                label = example["target"]
                if enforce_diversity and label_counts.get(label, 0) >= MAX_PER_LABEL:
                    continue
                part = f"Q: {example.get('raw_question', example['input'])}\nA: {label}"
                current_chars = sum(
                    len(f"Q: {ex.get('raw_question', ex['input'])}\nA: {ex['target']}") + 2
                    for _, ex in selected
                )
                if current_chars + len(part) > MAX_EXAMPLE_CHARS:
                    continue
                selected.append((score, example))
                used.add(index)
                label_counts[label] = label_counts.get(label, 0) + 1
                if len(selected) >= MAX_EXAMPLES:
                    return selected
        return selected

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._select(input)
        examples = "\n\n".join(
            f"Q: {ex.get('raw_question', ex['input'])}\nA: {ex['target']}"
            for _, ex in selected
        )
        self._last_context_length = len(examples)
        prompt = f"""Classify the problem below. Use the labeled examples as guidance, giving the closest matches more weight than distant ones. The examples may be incomplete; decide from the full input.

Relevant examples:
{examples if examples else '(No labeled examples yet.)'}

Problem:
{input}

Return only a JSON object with a concise string field named "final_answer" and an optional "reasoning" field."""
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(selected),
            "example_scores": [round(score, 4) for score, _ in selected],
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            example = {
                "input": str(result["input"]),
                "target": str(result["ground_truth"]),
            }
            if "raw_question" in result:
                example["raw_question"] = str(result["raw_question"])
            self.examples.append(example)

    def get_context_length(self) -> int:
        return self._last_context_length

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state or "{}")
        self.examples = list(data.get("examples", []))
        self._last_context_length = 0
