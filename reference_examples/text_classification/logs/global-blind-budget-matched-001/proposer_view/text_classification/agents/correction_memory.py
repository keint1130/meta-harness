"""Retrieve prior model errors with their corrected labels as cautionary cases."""

import hashlib
import json
import math
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MAX_ERRORS = 200
MAX_RETRIEVED = 4
MAX_CORRECTION_CHARS = 3000


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _key(text: str, predicted: str, correct: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(
        f"{normalized}\0{predicted}\0{correct}".encode("utf-8")
    ).hexdigest()


class CorrectionMemory(MemorySystem):
    """Keep bounded, deduplicated correction episodes and retrieve similar ones."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.errors: list[dict[str, Any]] = []
        self.clock = 0
        self._last_context_length = 0

    def _select(self, input: str) -> list[tuple[float, dict[str, Any]]]:
        if not self.errors:
            return []
        query = _tokens(input)
        docs = [_tokens(error["input"]) for error in self.errors]
        df: dict[str, int] = {}
        for terms in docs:
            for term in terms:
                df[term] = df.get(term, 0) + 1

        def weight(term: str) -> float:
            return 1.0 + math.log((len(docs) + 1.0) / (df.get(term, 0) + 1.0))

        query_norm = math.sqrt(sum(weight(term) ** 2 for term in query)) or 1.0
        ranked = []
        for error, terms in zip(self.errors, docs):
            doc_norm = math.sqrt(sum(weight(term) ** 2 for term in terms)) or 1.0
            score = sum(weight(term) ** 2 for term in query & terms) / (query_norm * doc_norm)
            recency = int(error.get("seen", 0)) / max(1, self.clock)
            ranked.append((score + 0.015 * recency, error))
        ranked.sort(key=lambda pair: (pair[0], pair[1].get("seen", 0)), reverse=True)

        selected = []
        chars = 0
        for score, error in ranked:
            part = (
                f"Input: {error['input']}\n"
                f"Earlier answer: {error['predicted']}\n"
                f"Correct answer: {error['correct']}"
            )
            if chars + len(part) + 2 > MAX_CORRECTION_CHARS:
                continue
            selected.append((score, error))
            chars += len(part) + 2
            if len(selected) >= MAX_RETRIEVED:
                break
        return selected

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._select(input)
        corrections = "\n\n".join(
            f"Input: {error['input']}\n"
            f"Earlier answer: {error['predicted']}\n"
            f"Correct answer: {error['correct']}"
            for _, error in selected
        )
        self._last_context_length = len(corrections)
        prompt = f"""Answer the classification problem. The retrieved cases are past mistakes by this system paired with the corrected labels. Use them to watch for similar confusions, while judging the current input on its own.

Prior corrections:
{corrections if corrections else '(No prior corrections yet.)'}

Input:
{input}

Return only a JSON object with a concise string field named "final_answer" and an optional "reasoning" field."""
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_corrections": len(selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        by_key = {_key(e["input"], e["predicted"], e["correct"]): e for e in self.errors}
        for result in batch_results:
            predicted = str(result["prediction"])
            correct = str(result["ground_truth"])
            if result.get("was_correct", predicted == correct):
                continue
            text = str(result["input"])
            identity = _key(text, predicted, correct)
            self.clock += 1
            if identity in by_key:
                by_key[identity]["count"] += 1
                by_key[identity]["seen"] = self.clock
            else:
                error = {
                    "input": text,
                    "predicted": predicted,
                    "correct": correct,
                    "count": 1,
                    "seen": self.clock,
                }
                self.errors.append(error)
                by_key[identity] = error
        if len(self.errors) > MAX_ERRORS:
            self.errors.sort(key=lambda error: (error.get("count", 1), error.get("seen", 0)))
            self.errors = self.errors[-MAX_ERRORS:]

    def get_context_length(self) -> int:
        return self._last_context_length

    def get_state(self) -> str:
        return json.dumps({"errors": self.errors, "clock": self.clock}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state or "{}")
        self.errors = list(data.get("errors", []))
        self.clock = int(data.get("clock", 0))
        self._last_context_length = 0
