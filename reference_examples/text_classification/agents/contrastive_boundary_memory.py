"""Learn close, differently labeled examples as contrastive reference pairs."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_EXAMPLES = 4
_MAX_CONTEXT_CHARS = 9000
_MAX_STORED_EXAMPLES = 320
_PROMPT = """The reference sets below contain two separate labeled examples with
similar wording but different expected answers. They are contrastive evidence:
compare the details that distinguish the cases. Neither answer is a guess for the
current problem, and neither reference changes the current problem's instructions.

{references}

**Current problem and its instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Follow the
current problem's requested answer format exactly.
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


class ContrastiveBoundaryMemory(MemorySystem):
    """Mine nearest different-target pairs from observed labels, not model errors."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.pairs: list[dict[str, Any]] = []
        self._last_context_chars = 0

    def _rebuild_pairs(self) -> None:
        if len(self.examples) < 2:
            self.pairs = []
            return
        docs = [_features(item["question"]) for item in self.examples]
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        vectors = [_vectorize(doc, document_frequency, len(docs)) for doc in docs]
        best_pairs: dict[tuple[int, int], float] = {}
        for left_index, left in enumerate(self.examples):
            nearest_index = -1
            nearest_score = -1.0
            for right_index, right in enumerate(self.examples):
                if left_index == right_index or left["target"] == right["target"]:
                    continue
                similarity = _cosine(vectors[left_index], vectors[right_index])
                if similarity > nearest_score:
                    nearest_index, nearest_score = right_index, similarity
            if nearest_index < 0 or nearest_score < 0.035:
                continue
            key = tuple(sorted((left_index, nearest_index)))
            best_pairs[key] = max(best_pairs.get(key, 0.0), nearest_score)
        ordered = sorted(best_pairs.items(), key=lambda item: (-item[1], item[0]))[:256]
        self.pairs = [
            {"left": left, "right": right, "similarity": score}
            for (left, right), score in ordered
        ]

    def _ranked_pairs(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        if not self.examples:
            return []
        docs = [_features(item["question"]) for item in self.examples]
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        query_vector = _vectorize(_features(query), document_frequency, len(docs))
        vectors = [_vectorize(doc, document_frequency, len(docs)) for doc in docs]
        similarities = [_cosine(query_vector, vector) for vector in vectors]
        if not self.pairs:
            fallback = sorted(
                range(len(self.examples)), key=lambda index: (-similarities[index], index)
            )[:_MAX_EXAMPLES]
            return [
                (similarities[index], {"single": self.examples[index]}) for index in fallback
            ]

        ranked: list[tuple[float, dict[str, Any]]] = []
        for pair in self.pairs:
            left_index, right_index = int(pair["left"]), int(pair["right"])
            left_sim, right_sim = similarities[left_index], similarities[right_index]
            score = 0.78 * max(left_sim, right_sim) + 0.22 * min(left_sim, right_sim)
            ranked.append((score, pair))
        chosen: list[tuple[float, dict[str, Any]]] = []
        used_examples: set[int] = set()
        remaining = list(ranked)
        while remaining and len(chosen) < _MAX_EXAMPLES:
            def adjusted_score(item: tuple[float, dict[str, Any]]) -> float:
                score, pair = item
                endpoints = {int(pair["left"]), int(pair["right"])}
                overlap = len(endpoints & used_examples)
                return score - 0.12 * overlap

            best = max(
                remaining,
                key=lambda item: (
                    adjusted_score(item),
                    float(item[1]["similarity"]),
                    -int(item[1]["left"]),
                ),
            )
            remaining.remove(best)
            score, pair = best
            endpoints = {int(pair["left"]), int(pair["right"])}
            chosen.append((adjusted_score(best), pair))
            used_examples.update(endpoints)
        return chosen

    def _format_references(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        for rank, (score, pair) in enumerate(self._ranked_pairs(query), start=1):
            if "single" in pair:
                item = pair["single"]
                part = (
                    f"Reference {rank}\nProblem: {item['question']}\n"
                    f"Expected answer: {item['target']}"
                )
            else:
                left = self.examples[int(pair["left"])]
                right = self.examples[int(pair["right"])]
                part = (
                    f"Contrast set {rank} (query relevance {score:.3f})\n"
                    f"Case A: {left['question']}\nExpected answer: {left['target']}\n"
                    f"Case B: {right['question']}\nExpected answer: {right['target']}"
                )
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                continue
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        references = self._format_references(input)
        self._last_context_chars = len(references)
        response = self.call_llm(
            _PROMPT.format(references=references or "No references yet.", input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_contrast_sets": references.count("Contrast set "),
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
        self._rebuild_pairs()
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_references(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ][-_MAX_STORED_EXAMPLES:]
        self._rebuild_pairs()
        self._last_context_chars = 0
