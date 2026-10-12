"""Rank and present only example content outside query-shared template lines."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_EXAMPLES = 8
_MAX_CONTEXT_CHARS = 10000
_MAX_STORED_EXAMPLES = 512
_PROMPT = """The references show the parts of earlier problems that differed from
their shared wording, together with their expected answers. Use those details as
evidence when relevant. The full current problem and its instructions govern.

{examples}

**Current problem and its instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Follow the
current problem's exact answer format.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


def _shared_line_residual(query: str, example: str) -> tuple[str, str]:
    left, right = query.splitlines(), example.splitlines()
    li = ri = 0
    limit = 0.55 * min(len(query), len(example))
    removed = 0
    while li < len(left) and ri < len(right) and left[li].strip() == right[ri].strip():
        line_size = max(len(left[li]), len(right[ri])) + 1
        if removed + line_size > limit:
            break
        removed += line_size
        li += 1
        ri += 1
    lend, rend = len(left), len(right)
    while lend > li and rend > ri and left[lend - 1].strip() == right[rend - 1].strip():
        line_size = max(len(left[lend - 1]), len(right[rend - 1])) + 1
        if removed + line_size > limit:
            break
        removed += line_size
        lend -= 1
        rend -= 1
    query_residual = "\n".join(left[li:lend]).strip()
    example_residual = "\n".join(right[ri:rend]).strip()
    if len(query_residual) < 24 or len(example_residual) < 24:
        return query, example
    return query_residual, example_residual


def _features(text: str) -> Counter[str]:
    lowered = text.lower()
    counts: Counter[str] = Counter("w:" + token for token in _WORD_RE.findall(lowered))
    for run in _RUN_RE.findall(lowered):
        for size in (2, 3):
            counts.update("c:" + run[index : index + size] for index in range(len(run) - size + 1))
    return counts


def _vectorize(features: Counter[str], df: Counter[str], total: int) -> dict[str, float]:
    values = {
        key: (1.0 + math.log(count))
        * (math.log((total + 1.0) / (df.get(key, 0) + 1.0)) + 1.0)
        for key, count in features.items()
    }
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {key: value / norm for key, value in values.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class SharedTemplateResidualMemory(MemorySystem):
    """Use pairwise shared-line removal to expose content beneath prompt boilerplate."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _ranked_examples(self, query: str) -> list[tuple[float, int, str]]:
        if not self.examples:
            return []
        residuals = [
            _shared_line_residual(query, item["question"])[1] for item in self.examples
        ]
        query_residuals = [
            _shared_line_residual(query, item["question"])[0] for item in self.examples
        ]
        docs = [_features(text) for text in residuals]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        vectors = [_vectorize(doc, df, len(docs)) for doc in docs]
        ranked: list[tuple[float, int, str]] = []
        for index, query_text in enumerate(query_residuals):
            qvec = _vectorize(_features(query_text), df, len(docs))
            ranked.append((_cosine(qvec, vectors[index]), index, residuals[index]))
        return sorted(ranked, key=lambda row: (-row[0], row[1]))[:_MAX_EXAMPLES]

    def _format_examples(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        for rank, (score, index, residual) in enumerate(self._ranked_examples(query), start=1):
            item = self.examples[index]
            part = (
                f"Reference {rank} (residual relevance {score:.3f})\n"
                f"Distinguishing problem content: {residual}\n"
                f"Expected answer: {item['target']}"
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
        seen = {(item["question"], item["target"]) for item in self.examples}
        for result in batch_results:
            item = {
                "question": _question_from_result(result),
                "target": str(result.get("ground_truth", "")),
            }
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-_MAX_STORED_EXAMPLES:]
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
        ][-_MAX_STORED_EXAMPLES:]
        self._last_context_chars = 0
