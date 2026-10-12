"""Retrieve relevant examples while avoiding redundant answer patterns."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PROMPT = """Use the labeled references as evidence. References are selected for
similarity and variety, so compare their specific details instead of assuming
that a nearby example has the same answer. Follow the live problem's content
and instructions as authoritative.

{examples}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. The
final_answer must follow the current problem's requested format.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _features(text: str) -> Counter[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    features: Counter[str] = Counter("w:" + token for token in tokens)
    for token in tokens:
        features.update("c:" + token[i : i + 3] for i in range(max(0, len(token) - 2)))
    return features


def _cosine(left: Counter[str], right: Counter[str], df: Counter[str], total: int) -> float:
    def weights(features: Counter[str]) -> dict[str, float]:
        values = {key: (1 + math.log(n)) * (1 + math.log((total + 1) / (df[key] + 1))) for key, n in features.items()}
        norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
        return {key: value / norm for key, value in values.items()}
    a, b = weights(left), weights(right)
    return sum(value * b.get(key, 0.0) for key, value in a.items())


class QueryDiverseRetrieval(MemorySystem):
    """Select close references, penalizing repeated answer labels modestly."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _rank(self, query: str) -> list[tuple[float, int, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_features(item["question"]) for item in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        q = _features(query)
        scores = [_cosine(q, doc, df, len(docs)) for doc in docs]
        chosen: list[tuple[float, int, dict[str, str]]] = []
        used: set[int] = set()
        answer_counts: Counter[str] = Counter()
        while len(chosen) < min(6, len(self.examples)):
            options = []
            for i, item in enumerate(self.examples):
                if i in used:
                    continue
                # Retain relevance as the main signal; use answer repetition only as a tie-break-like discount.
                adjusted = scores[i] / (1.0 + 0.22 * answer_counts[item["target"]])
                options.append((adjusted, scores[i], -i, i, item))
            _, score, _, index, item = max(options)
            used.add(index)
            answer_counts[item["target"]] += 1
            chosen.append((score, index, item))
        return chosen

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = []
        for rank, (score, _, item) in enumerate(self._rank(input), 1):
            parts.append(f"Reference {rank} (similarity {score:.3f})\nProblem: {item['question']}\nExpected answer: {item['target']}")
        examples = "\n\n".join(parts) or "No references yet."
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "num_examples": len(parts)}

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
        return self._last_context_chars or sum(len(x["question"]) + len(x["target"]) + 34 for x in self.examples[-6:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
