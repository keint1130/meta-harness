"""Retrieve examples using both content similarity and output-structure overlap."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TAG_RE = re.compile(r"\[[^\]\r\n]{1,48}\]|<[^>\r\n]{1,48}>")
_KEY_RE = re.compile(r'"([A-Za-z_][\w -]{0,30})"\s*:')
_CUE_RE = re.compile(r"\b(format|output|answer|respond|return|exact|serialize|delimiter|separated|must)\b", re.I)
_PROMPT = """The references are examples selected for both problem content and
similar answer structure. Use them as evidence while following the full current
problem and its answer requirements.

{examples}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Preserve the
answer shape requested in the current problem.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _terms(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _WORD_RE.findall(text))


def _shape(text: str) -> set[str]:
    result = {"tag:" + x.lower() for x in _TAG_RE.findall(text)}
    result.update("key:" + x.lower() for x in _KEY_RE.findall(text))
    for line in text.splitlines():
        if _CUE_RE.search(line):
            words = [x.lower() for x in _WORD_RE.findall(line) if x.isalnum()]
            if words:
                result.add("cue:" + " ".join(words[:18]))
    return result


def _cosine(left: Counter[str], right: Counter[str], df: Counter[str], n: int) -> float:
    def vector(features: Counter[str]) -> dict[str, float]:
        vals = {k: (1 + math.log(v)) * (1 + math.log((n + 1) / (df[k] + 1))) for k, v in features.items()}
        norm = math.sqrt(sum(x * x for x in vals.values())) or 1.0
        return {k: v / norm for k, v in vals.items()}
    a, b = vector(left), vector(right)
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class SchemaSignatureRetrieval(MemorySystem):
    """Blend lexical neighbors with overlap in generic answer-format markers."""

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
        qterms, qshape = _terms(query), _shape(query)
        ranked = []
        for item, doc in zip(self.examples, docs):
            lexical = _cosine(qterms, doc, df, len(docs))
            dshape = _shape(item["question"])
            overlap = len(qshape & dshape) / max(1, len(qshape | dshape))
            ranked.append((0.78 * lexical + 0.22 * overlap, item))
        return sorted(ranked, key=lambda pair: -pair[0])[:5]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank(input)
        examples = "\n\n".join(
            f"Reference {i} (content and structure relevance {score:.3f})\nProblem: {item['question']}\nExpected answer: {item['target']}"
            for i, (score, item) in enumerate(selected, 1)
        ) or "No references yet."
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(selected),
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
        return self._last_context_chars or sum(len(x["question"]) + len(x["target"]) + 42 for x in self.examples[-5:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
