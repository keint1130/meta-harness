"""Query-adaptive retrieval of diverse, label-contrasting demonstrations."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem using the labeled examples as contrasts.

{examples_section}

**Problem:**
{input}

Compare the complete pattern in the problem with the examples. Pay attention to
how examples with different labels differ, and do not copy a label based on one
shared detail. Give the best supported answer in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 12000
MAX_SELECTED = 5


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]{2,}", text.lower())


class ContrastiveRetrieval(MemorySystem):
    """Retrieve query-relevant demonstrations while reducing label redundancy."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    @staticmethod
    def _example_text(example: dict[str, str]) -> str:
        return example.get("raw_question") or example["input"]

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _select_examples(self, query: str) -> list[dict[str, str]]:
        if not self.examples:
            return []

        doc_freq: Counter[str] = Counter()
        for example in self.examples:
            doc_freq.update(set(_tokens(self._example_text(example))))
        total = len(self.examples)

        def vector(text: str) -> dict[str, float]:
            counts = Counter(_tokens(text))
            return {
                token: count * (math.log((total + 1) / (doc_freq[token] + 1)) + 1)
                for token, count in counts.items()
            }

        vectors = [vector(self._example_text(example)) for example in self.examples]
        query_vector = vector(query)
        relevance = [self._cosine(query_vector, item) for item in vectors]
        remaining = set(range(total))
        selected: list[int] = []
        selected_labels: set[str] = set()

        while remaining and len(selected) < MAX_SELECTED:
            def rank(index: int) -> float:
                redundancy = max(
                    (self._cosine(vectors[index], vectors[chosen]) for chosen in selected),
                    default=0.0,
                )
                label_bonus = 0.08 if self.examples[index]["target"] not in selected_labels else 0.0
                return relevance[index] + label_bonus - 0.30 * redundancy

            best = max(remaining, key=rank)
            selected.append(best)
            selected_labels.add(self.examples[best]["target"])
            remaining.remove(best)

        selected.sort(key=lambda index: relevance[index], reverse=True)
        chosen = []
        chars = 0
        for index in selected:
            example = self.examples[index]
            question = self._example_text(example)
            part = f"Case: {question}\nLabel: {example['target']}"
            if chars + len(part) > MAX_CHARS:
                continue
            chosen.append(example)
            chars += len(part) + 2
        return chosen

    def _format_examples_section(self, query: str = "") -> str:
        selected = self._select_examples(query)
        if not selected:
            return ""
        return "Contrasting labeled cases:\n\n" + "\n\n".join(
            f"Case: {self._example_text(example)}\nLabel: {example['target']}"
            for example in selected
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._select_examples(input)
        examples_section = "Contrasting labeled cases:\n\n" + "\n\n".join(
            f"Case: {self._example_text(example)}\nLabel: {example['target']}"
            for example in selected
        ) if selected else ""
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(selected),
            "num_distinct_labels": len({example["target"] for example in selected}),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            example = {"input": result["input"], "target": result["ground_truth"]}
            if "raw_question" in result:
                example["raw_question"] = result["raw_question"]
            self.examples.append(example)

    def get_context_length(self) -> int:
        return len(self._format_examples_section())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
