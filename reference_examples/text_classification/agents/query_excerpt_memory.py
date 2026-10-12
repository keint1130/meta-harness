"""Retrieve labeled cases and quote only their query-relevant text excerpts."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_PROMPT = """The references quote selected passages from earlier problems, with
their expected answers. These excerpts may omit context, so use them only when
their details match the complete current problem. The current instructions govern.

{examples}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields, following
the current problem's answer format.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _terms(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN_RE.findall(text))


def _vector(features: Counter[str], df: Counter[str], n: int) -> dict[str, float]:
    values = {key: (1 + math.log(count)) * (1 + math.log((n + 1) / (df[key] + 1))) for key, count in features.items()}
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {key: value / norm for key, value in values.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class QueryExcerptMemory(MemorySystem):
    """Make long retrieved examples compact by selecting high-overlap passages."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _rank(self, query: str) -> list[tuple[float, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_terms(item["question"]) for item in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        qvec = _vector(_terms(query), df, len(docs))
        return sorted(
            (( _cosine(qvec, _vector(doc, df, len(docs))), item) for item, doc in zip(self.examples, docs)),
            key=lambda pair: -pair[0],
        )[:5]

    def _excerpt(self, query: str, question: str, df: Counter[str], n: int) -> str:
        query_terms = _terms(query)
        pieces = [part.strip() for part in _SENTENCE_RE.split(question) if len(part.strip()) >= 18]
        if not pieces:
            return question[:320]
        scored = []
        for index, piece in enumerate(pieces):
            terms = _terms(piece)
            overlap = sum(
                min(count, 2)
                * query_terms[term]
                * math.log((n + 1) / (df[term] + 1))
                / (1 + math.log(1 + count))
                for term, count in terms.items()
                if term in query_terms and df[term] < n
            )
            scored.append((overlap / math.sqrt(len(terms) + 1), -index, piece))
        chosen = sorted(scored, reverse=True)[:2]
        if not chosen or chosen[0][0] <= 0:
            return question[:320]
        excerpt = " … ".join(piece for _, _, piece in chosen)
        return excerpt[:700]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank(input)
        doc_df: Counter[str] = Counter()
        for item in self.examples:
            doc_df.update(_terms(item["question"]).keys())
        parts = [
            f"Reference {i} (similarity {score:.3f})\nRelevant passage: {self._excerpt(input, item['question'], doc_df, len(self.examples))}\nExpected answer: {item['target']}"
            for i, (score, item) in enumerate(selected, 1)
        ]
        examples = "\n\n".join(parts) or "No references yet."
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(parts),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["question"], x["target"]) for x in self.examples}
        for result in batch_results:
            item = {"question": _question(result), "target": str(result.get("ground_truth", ""))}
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-512:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or sum(min(len(x["question"]), 700) + len(x["target"]) + 42 for x in self.examples[-5:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
