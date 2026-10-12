"""Adapt example-set breadth to the strength of the nearest lexical match."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PROMPT = """The references are labeled examples with different degrees of
relevance. Treat close matches as stronger evidence and broad matches only as
background. Similar wording does not replace checking the current details.

{examples}

**Current problem and its instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Follow the
current problem's answer format.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _features(text: str) -> Counter[str]:
    tokens = [x.lower() for x in _TOKEN_RE.findall(text)]
    result: Counter[str] = Counter("w:" + x for x in tokens)
    for token in tokens:
        result.update("c:" + token[i:i+3] for i in range(max(0, len(token) - 2)))
    return result


def _vector(features: Counter[str], df: Counter[str], n: int) -> dict[str, float]:
    raw = {k: (1 + math.log(c)) * (1 + math.log((n + 1) / (df[k] + 1))) for k, c in features.items()}
    norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
    return {k: v / norm for k, v in raw.items()}


class ConfidenceAdaptiveMemory(MemorySystem):
    """Use a narrow neighborhood for confident matches and broaden uncertain ones."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _ranked(self, query: str) -> list[tuple[float, int, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_features(x["question"]) for x in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        q = _vector(_features(query), df, len(docs))
        ranked = []
        for i, (item, doc) in enumerate(zip(self.examples, docs)):
            d = _vector(doc, df, len(docs))
            ranked.append((sum(v * d.get(k, 0.0) for k, v in q.items()), i, item))
        return sorted(ranked, key=lambda x: (-x[0], x[1]))

    def _select(self, query: str) -> list[tuple[float, int, dict[str, str]]]:
        ranked = self._ranked(query)
        if not ranked:
            return []
        top = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = top - runner_up
        # Scores are normalized cosine similarities; a weak or ambiguous top match
        # triggers a broader evidence set. Strong matches keep context compact.
        limit = 4 if top >= 0.32 and margin >= 0.06 else (8 if top >= 0.18 else 12)
        return ranked[:limit]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._select(input)
        parts = [f"Reference {i} (similarity {score:.3f})\nProblem: {item['question']}\nExpected answer: {item['target']}" for i, (score, _, item) in enumerate(selected, 1)]
        examples = "\n\n".join(parts) or "No references yet."
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        top_score = selected[0][0] if selected else 0.0
        return extract_json_field(response, "final_answer"), {"full_response": response, "num_examples": len(parts), "top_similarity": top_score}

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
        return self._last_context_chars or sum(len(x["question"]) + len(x["target"]) + 36 for x in self.examples[-12:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
