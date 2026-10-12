"""Compact answer-pattern profiles learned from labeled examples."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The learned answer patterns below summarize feedback from prior examples. Use them as candidate guidance, then inspect the current problem and follow its instructions. An answer may contain one or several output labels; do not omit a supported part.

{profiles}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]"}}"""


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


def _clip(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


class Candidate(MemorySystem):
    """Represent training feedback as answer profiles with supporting cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_len = 0
        self.profile_budget = 8500

    def _profiles(self, query: str) -> str:
        if not self.examples:
            self._last_context_len = 0
            return "No learned answer patterns yet."

        docs = [_features(row["input"]) for row in self.examples]
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        count = len(docs)
        idf = {
            token: math.log(1.0 + (count + 1) / (frequency + 1))
            for token, frequency in document_frequency.items()
        }

        doc_vectors: list[dict[str, float]] = []
        query_counts = _features(query)
        query_vector = {
            token: value * idf.get(token, 0.0) for token, value in query_counts.items()
        }
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        query_vector = {token: value / query_norm for token, value in query_vector.items()}

        class_vectors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        class_counts: Counter[str] = Counter()
        for row, doc in zip(self.examples, docs):
            raw_total = sum(doc.values()) or 1
            vector = {
                token: value / raw_total * idf.get(token, 0.0)
                for token, value in doc.items()
            }
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            vector = {token: value / norm for token, value in vector.items()}
            doc_vectors.append(vector)
            label = row["target"]
            class_counts[label] += 1
            for token, value in vector.items():
                class_vectors[label][token] += value

        prototypes: dict[str, dict[str, float]] = {}
        for label, vector in class_vectors.items():
            averaged = {token: value / class_counts[label] for token, value in vector.items()}
            norm = math.sqrt(sum(value * value for value in averaged.values())) or 1.0
            prototypes[label] = {token: value / norm for token, value in averaged.items()}

        class_scores = []
        for label, vector in prototypes.items():
            score = sum(query_vector.get(token, 0.0) * value for token, value in vector.items())
            class_scores.append((score, label))
        class_scores.sort(key=lambda item: (item[0], class_counts[item[1]], item[1]), reverse=True)

        cards: list[str] = []
        used = 0
        for score, label in class_scores[:8]:
            best_index = max(
                (index for index, row in enumerate(self.examples) if row["target"] == label),
                key=lambda index: sum(
                    query_vector.get(token, 0.0) * value
                    for token, value in doc_vectors[index].items()
                ),
            )
            support = _clip(self.examples[best_index]["input"])
            card = (
                f"Observed answer pattern: {label} "
                f"(support={class_counts[label]}, profile_match={score:.3f})\n"
                f"Closest supporting case: {support}"
            )
            if used + len(card) > self.profile_budget:
                continue
            cards.append(card)
            used += len(card) + 2
        self._last_context_len = used
        return "\n\n".join(cards) or "No matching answer profiles."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles = self._profiles(input)
        response = self.call_llm(_PROMPT.format(profiles=profiles, input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "answer_patterns": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            self.examples.append(
                {"input": _source_text(row), "target": str(row["ground_truth"])}
            )

    def get_context_length(self) -> int:
        return self._last_context_len

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])

