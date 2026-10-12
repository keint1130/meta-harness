"""Select few-shot examples using query-specific lexical relevance."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_CHAR_RUN_RE = re.compile(r"[\w#@+\-/=().\[\]]+", re.UNICODE)
_PROMPT = """Use the reference examples as evidence for solving the current problem.
Choose the references most relevant to the current problem. Follow the current
problem's instructions for its answer format.

{examples}

**Current problem:**
{input}

Return a JSON object with a concise \"final_answer\" field and a \"reasoning\" field.
"""


def _features(text: str) -> Counter[str]:
    lowered = text.lower()
    counts: Counter[str] = Counter("w:" + token for token in _WORD_RE.findall(lowered))
    for run in _CHAR_RUN_RE.findall(lowered):
        compact = re.sub(r"\s+", "", run)
        for size in (2, 3):
            counts.update("c:" + compact[i : i + size] for i in range(len(compact) - size + 1))
    return counts


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


class RetrievalWeightedExamples(MemorySystem):
    """Inject a small set of examples ranked by inverse-frequency overlap."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 6
        self.max_context_chars = 12000
        self._last_context_chars = 0

    def _ranked_examples(self, query: str) -> list[tuple[float, int, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_features(item["question"]) for item in self.examples]
        query_features = _features(query)
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        total = len(docs)

        def weighted(features: Counter[str]) -> dict[str, float]:
            values = {
                key: (1.0 + math.log(count))
                * (math.log((total + 1.0) / (document_frequency.get(key, 0) + 1.0)) + 1.0)
                for key, count in features.items()
            }
            norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
            return {key: value / norm for key, value in values.items()}

        query_vector = weighted(query_features)
        ranked = []
        for index, (item, features) in enumerate(zip(self.examples, docs)):
            vector = weighted(features)
            score = sum(query_vector.get(key, 0.0) * value for key, value in vector.items())
            ranked.append((score, index, item))
        return sorted(ranked, key=lambda row: (-row[0], row[1]))

    def _format_examples(self, query: str) -> str:
        selected = self._ranked_examples(query)[: self.max_examples]
        parts: list[str] = []
        used = 0
        for rank, (score, _, item) in enumerate(selected, start=1):
            part = (
                f"Reference {rank} (similarity {score:.3f})\n"
                f"Problem: {item['question']}\nExpected answer: {item['target']}"
            )
            added = len(part) + (2 if parts else 0)
            if used + added > self.max_context_chars:
                continue
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples or "No references yet.", input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_selected_examples": examples.count("Reference ")}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "question": _question_from_result(result),
                    "target": str(result.get("ground_truth", "")),
                }
            )

    def get_context_length(self) -> int:
        if self._last_context_chars:
            return self._last_context_chars
        return len(self._format_examples(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ]
        self._last_context_chars = 0
