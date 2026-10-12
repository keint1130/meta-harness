"""Retrieve focused references while discouraging redundant neighbors."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HEADING_RE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+[^\n]*")
_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_EXAMPLES = 7
_MAX_CONTEXT_CHARS = 9000
_PROMPT = """Use the references as examples of useful patterns. Rank their relevance
to the facts in the current problem, but treat the current problem's instructions
and output format as authoritative. Similar wording alone does not determine the
answer.

{examples}

**Current problem:**
{input}

Return a JSON object with "reasoning" and "final_answer" fields. Make the
final_answer concise and obey the current problem's exact answer format.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


def _features(text: str) -> Counter[str]:
    lowered = text.lower()
    counts: Counter[str] = Counter("w:" + token for token in _WORD_RE.findall(lowered))
    for run in _RUN_RE.findall(lowered):
        compact = "".join(ch for ch in run if ch.isalnum())
        for size in (2, 3):
            counts.update("c:" + compact[i : i + size] for i in range(len(compact) - size + 1))
    return counts


def _focus_text(text: str) -> str:
    headings = list(_HEADING_RE.finditer(text))
    if headings:
        focus = text[headings[-1].end() :].strip()
        if focus:
            return focus
    return text


def _query_features(text: str) -> Counter[str]:
    result = _features(text)
    result.update(_features(_focus_text(text)))
    result.update(_features(_focus_text(text)))
    return result


def _vectorize(
    features: Counter[str], document_frequency: Counter[str], total_docs: int
) -> dict[str, float]:
    values = {
        key: (1.0 + math.log(count))
        * (math.log((total_docs + 1.0) / (document_frequency.get(key, 0) + 1.0)) + 1.0)
        for key, count in features.items()
    }
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {key: value / norm for key, value in values.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class FocusMMRRetrieval(MemorySystem):
    """Give more weight to the last headed content section and diversify selection."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _ranked_examples(self, query: str) -> list[tuple[float, int, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_features(item["question"]) for item in self.examples]
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        query_vector = _vectorize(_query_features(query), document_frequency, len(docs))
        vectors = [_vectorize(doc, document_frequency, len(docs)) for doc in docs]
        relevance = [_cosine(query_vector, vector) for vector in vectors]

        chosen: list[int] = []
        remaining = set(range(len(self.examples)))
        while remaining and len(chosen) < _MAX_EXAMPLES:
            def score(index: int) -> float:
                redundancy = max((_cosine(vectors[index], vectors[prior]) for prior in chosen), default=0.0)
                repeated_target = any(
                    self.examples[prior]["target"] == self.examples[index]["target"]
                    for prior in chosen
                )
                return 0.82 * relevance[index] - 0.18 * redundancy - (0.06 if repeated_target else 0.0)

            best = max(remaining, key=lambda index: (score(index), -index))
            chosen.append(best)
            remaining.remove(best)
        return [(relevance[index], index, self.examples[index]) for index in chosen]

    def _format_examples(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        for rank, (similarity, _, item) in enumerate(self._ranked_examples(query), start=1):
            part = (
                f"Reference {rank} (relevance {similarity:.3f})\n"
                f"Problem: {item['question']}\nExpected answer: {item['target']}"
            )
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                continue
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        self._last_context_chars = len(examples)
        response = self.call_llm(
            _PROMPT.format(examples=examples or "No references yet.", input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_selected_examples": examples.count("Reference "),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "question": _question_from_result(result),
                    "target": str(result.get("ground_truth", "")),
                }
            )
        self.examples = self.examples[-512:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_examples(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ][-512:]
        self._last_context_chars = 0
