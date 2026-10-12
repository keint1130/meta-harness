"""Retrieve labeled cases by blending query similarity with graph walks."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using these directly relevant and graph-related cases.

{examples_section}

**Problem:**
{input}

Nearby cases can connect to other useful precedents through shared details.
Use those connections to find a pattern, then verify it against the complete
problem before answering. Respond in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

TOP_K = 5
GRAPH_NEIGHBORS = 6
WALK_STEPS = 5
RESTART_PROBABILITY = 0.30
DIRECT_WEIGHT = 0.90
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _char_ngrams(text: str) -> Counter[str]:
    features: Counter[str] = Counter()
    for chunk in _WORD_RE.findall(text.lower()):
        for size in (2, 3):
            if len(chunk) >= size:
                features.update(
                    chunk[index : index + size]
                    for index in range(len(chunk) - size + 1)
                )
    return features


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


class GraphWalkMemory(MemorySystem):
    """Use a case-neighbor graph to propagate query relevance between examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    @staticmethod
    def _text(example: dict[str, str]) -> str:
        return example.get("raw_question") or example["input"]

    def _rank_examples(self, query: str) -> list[tuple[float, int, float]]:
        count = len(self.examples)
        if not count:
            return []

        document_features = [_char_ngrams(self._text(example)) for example in self.examples]
        query_features = _char_ngrams(query)
        document_frequency: Counter[str] = Counter()
        for features in document_features:
            document_frequency.update(features.keys())
        vectors = []
        for features in document_features:
            weighted = {
                feature: frequency
                * (math.log((count + 1) / (document_frequency[feature] + 1)) + 1.0)
                for feature, frequency in features.items()
            }
            norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
            vectors.append({feature: value / norm for feature, value in weighted.items()})
        weighted_query = {
            feature: frequency
            * (math.log((count + 1) / (document_frequency[feature] + 1)) + 1.0)
            for feature, frequency in query_features.items()
        }
        query_norm = math.sqrt(sum(value * value for value in weighted_query.values())) or 1.0
        query_vector = {feature: value / query_norm for feature, value in weighted_query.items()}

        direct = [_cosine(query_vector, vector) for vector in vectors]
        direct_total = sum(max(0.0, score) for score in direct) or 1.0
        seed = [max(0.0, score) / direct_total for score in direct]

        similarities = [
            [_cosine(vectors[index], vectors[other]) if index != other else 0.0
             for other in range(count)]
            for index in range(count)
        ]
        transitions: list[list[tuple[int, float]]] = []
        for index, row in enumerate(similarities):
            neighbors = sorted(
                (other for other in range(count) if other != index),
                key=lambda other: row[other],
                reverse=True,
            )[:GRAPH_NEIGHBORS]
            total_weight = sum(max(0.0, row[other]) for other in neighbors) or 1.0
            transitions.append(
                [
                    (other, max(0.0, row[other]) / total_weight)
                    for other in neighbors
                    if row[other] > 0
                ]
            )

        walk = seed[:]
        for _ in range(WALK_STEPS):
            updated = [RESTART_PROBABILITY * seed[index] for index in range(count)]
            for source, edges in enumerate(transitions):
                for destination, probability in edges:
                    updated[destination] += (
                        (1.0 - RESTART_PROBABILITY) * walk[source] * probability
                    )
            walk = updated

        ranked = [
            (
                DIRECT_WEIGHT * seed[index] + (1.0 - DIRECT_WEIGHT) * walk[index],
                index,
                walk[index],
            )
            for index in range(count)
        ]
        ranked.sort(reverse=True)
        return ranked[:TOP_K]

    def _format_examples(self, query: str) -> str:
        selected = self._rank_examples(query)
        if not selected:
            return "No labeled cases have been learned yet."
        rows = []
        for rank, (_, index, walk_score) in enumerate(selected, start=1):
            example = self.examples[index]
            rows.append(
                f"Case {rank} (graph support {walk_score:.3f}):\n"
                f"Problem: {self._text(example)}\n"
                f"Correct output: {example['target']}"
            )
        return "Query-connected cases:\n\n" + "\n\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank_examples(input)
        response = self.call_llm(
            PROMPT_TEMPLATE.format(
                examples_section=self._format_examples(input), input=input
            )
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(selected),
            "mean_graph_support": (
                sum(item[2] for item in selected) / len(selected) if selected else 0.0
            ),
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
        return len(self._format_examples(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
