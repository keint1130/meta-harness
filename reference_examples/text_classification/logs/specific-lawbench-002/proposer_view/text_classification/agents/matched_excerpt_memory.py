"""Store labeled sentence fragments and retrieve evidence-sized excerpts."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The learned examples are partial text excerpts selected for overlap with the current problem. Use them as evidence, while remembering that each excerpt omits surrounding details. Follow the current problem's own instructions.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


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


def _split_fragments(text: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?；;])\s*|\r?\n+", text)
    fragments: list[str] = []
    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if len(sentence) < 35:
            continue
        if len(sentence) <= 620:
            fragments.append(sentence)
            continue
        # Long sentences become smaller evidence units without losing order.
        for piece in re.split(r"(?<=[，,])\s*", sentence):
            piece = piece.strip()
            if len(piece) >= 35:
                fragments.append(piece[:620])
    return fragments


def _clip(text: str, limit: int = 560) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


class Candidate(MemorySystem):
    """Represent examples as query-matchable fragments rather than whole cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.fragments: list[dict[str, Any]] = []
        self._next_source_id = 0
        self.context_budget = 7600
        self.max_sources = 12
        self._last_context = ""

    def _vectors(self) -> tuple[list[dict[str, float]], dict[str, float]]:
        documents = [_features(fragment["text"]) for fragment in self.fragments]
        frequency: Counter[str] = Counter()
        source_features: dict[int, set[str]] = {}
        for fragment, document in zip(self.fragments, documents):
            source_features.setdefault(fragment["source_id"], set()).update(document.keys())
        for tokens in source_features.values():
            frequency.update(tokens)
        count = len(source_features)
        inverse = {
            token: math.log(1.0 + (count + 1) / (value + 1))
            for token, value in frequency.items()
        }
        vectors: list[dict[str, float]] = []
        for document in documents:
            total = sum(document.values()) or 1
            raw = {token: value / total * inverse[token] for token, value in document.items()}
            norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
            vectors.append({token: value / norm for token, value in raw.items()})
        return vectors, inverse

    def _format_examples(self, query: str) -> str:
        if not self.fragments:
            self._last_context = ""
            return "No learned excerpts yet."
        vectors, inverse = self._vectors()
        counts = _features(query)
        raw = {token: value * inverse.get(token, 0.0) for token, value in counts.items()}
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
        query_vector = {token: value / norm for token, value in raw.items()}
        ranked = sorted(
            ((_cosine(query_vector, vector), index) for index, vector in enumerate(vectors)),
            reverse=True,
        )
        parts: list[str] = []
        used = 0
        sources: set[int] = set()
        for score, index in ranked:
            fragment = self.fragments[index]
            if score <= 0 or fragment["source_id"] in sources:
                continue
            part = f"Related excerpt: {_clip(fragment['text'])}\nVerified answer: {fragment['target']}"
            remaining = self.context_budget - used
            if remaining < 100:
                break
            if len(part) > remaining:
                part = part[: remaining - 3] + "..."
            parts.append(part)
            sources.add(fragment["source_id"])
            used += len(part) + 2
            if len(sources) >= self.max_sources:
                break
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No overlapping learned excerpts yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_excerpts": len(self._last_context.split("\n\n")) if self._last_context else 0,
            "memory_sources": self._next_source_id,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {
            (fragment["source_id"], fragment["text"])
            for fragment in self.fragments
        }
        for row in batch_results:
            text = _source_text(row)
            target = str(row["ground_truth"])
            source_id = self._next_source_id
            self._next_source_id += 1
            for fragment_text in _split_fragments(text)[:80]:
                key = (source_id, fragment_text)
                if key not in seen:
                    self.fragments.append(
                        {"source_id": source_id, "text": fragment_text, "target": target}
                    )
                    seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps(
            {"fragments": self.fragments, "next_source_id": self._next_source_id},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.fragments = data.get("fragments", [])
        self._next_source_id = data.get(
            "next_source_id",
            max((item.get("source_id", -1) for item in self.fragments), default=-1) + 1,
        )
        self._last_context = ""
