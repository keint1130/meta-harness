"""Rerank retrieved demonstrations using their linked prediction feedback."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_EXAMPLES = 6
_MAX_CONTEXT_CHARS = 10000
_MAX_STORED_EXAMPLES = 512
_PROMPT = """Use the references as evidence for solving the current problem. Their
selection reflects both text relevance and feedback from earlier predictions;
check the details before applying an answer pattern. The current problem's own
instructions determine its answer format.

{examples}

**Current problem:**
{input}

Return a JSON object with "reasoning" and "final_answer" fields. The final_answer
must follow the current problem's requested format.
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


class FeedbackWeightedRetrieval(MemorySystem):
    """Credit or downweight references that participated in prior predictions."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.next_id = 0
        self._last_context_chars = 0

    def _ranked_examples(self, query: str) -> list[tuple[float, int, dict[str, Any]]]:
        if not self.examples:
            return []
        docs = [_features(item["question"]) for item in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        qvec = _vectorize(_features(query), df, len(docs))
        ranked = []
        for index, (item, doc) in enumerate(zip(self.examples, docs)):
            similarity = _cosine(qvec, _vectorize(doc, df, len(docs)))
            wins = int(item.get("wins", 0))
            losses = int(item.get("losses", 0))
            reliability = (wins + 1.0) / (wins + losses + 2.0)
            feedback_factor = 0.75 + 0.5 * reliability
            ranked.append((similarity * feedback_factor, index, item))
        return sorted(ranked, key=lambda row: (-row[0], row[1]))[:_MAX_EXAMPLES]

    def _format_examples(self, query: str) -> tuple[str, list[int]]:
        parts: list[str] = []
        selected_ids: list[int] = []
        used = 0
        for rank, (score, _, item) in enumerate(self._ranked_examples(query), start=1):
            part = (
                f"Reference {rank} (relevance and feedback {score:.3f})\n"
                f"Problem: {item['question']}\nExpected answer: {item['target']}"
            )
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                continue
            parts.append(part)
            selected_ids.append(int(item["id"]))
            used += added
        return "\n\n".join(parts), selected_ids

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples, selected_ids = self._format_examples(input)
        self._last_context_chars = len(examples)
        response = self.call_llm(
            _PROMPT.format(examples=examples or "No references yet.", input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_selected_examples": len(selected_ids),
            "selected_example_ids": selected_ids,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        by_id = {int(item["id"]): item for item in self.examples}
        for result in batch_results:
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            outcome = bool(result.get("was_correct", False))
            for selected_id in metadata.get("selected_example_ids", []):
                item = by_id.get(int(selected_id))
                if item is None:
                    continue
                key = "wins" if outcome else "losses"
                item[key] = int(item.get(key, 0)) + 1
        seen = {(item["question"], item["target"]) for item in self.examples}
        for result in batch_results:
            item = {
                "id": self.next_id,
                "question": _question_from_result(result),
                "target": str(result.get("ground_truth", "")),
                "wins": 0,
                "losses": 0,
            }
            self.next_id += 1
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-_MAX_STORED_EXAMPLES:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_examples("")[0])

    def get_state(self) -> str:
        return json.dumps(
            {"examples": self.examples, "next_id": self.next_id}, ensure_ascii=False
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        loaded = data.get("examples", [])
        self.examples = []
        for position, item in enumerate(loaded[-_MAX_STORED_EXAMPLES:]):
            self.examples.append(
                {
                    "id": int(item.get("id", position)),
                    "question": str(item.get("question", "")),
                    "target": str(item.get("target", "")),
                    "wins": int(item.get("wins", 0)),
                    "losses": int(item.get("losses", 0)),
                }
            )
        self.next_id = int(
            data.get("next_id", max((int(item["id"]) for item in self.examples), default=-1) + 1)
        )
        self._last_context_chars = 0
