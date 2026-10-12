"""Use previously incorrect labeled cases as targeted correction demonstrations."""

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """The following cases record earlier mistakes and their corrected labels.
Treat each correction as authoritative evidence. Compare the current problem with the
closest corrected cases, then give one concise answer using the exact corrected label.
Follow any answer format explicitly requested in the problem, and do not add qualifications.

{corrections}

Problem:
{input}

Return JSON only: {{"reasoning": "brief rationale", "final_answer": "exact label"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his", "i",
    "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "she", "that", "the", "their", "them", "there", "they", "this",
    "to", "up", "was", "we", "were", "when", "which", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1 and token not in _STOP]


def _shared_prefix(texts: list[str]) -> str:
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        end = min(len(prefix), len(text))
        pos = 0
        while pos < end and prefix[pos] == text[pos]:
            pos += 1
        prefix = prefix[:pos]
    return prefix if len(prefix) >= 40 else ""


def _body(text: str, prefix: str) -> str:
    return text[len(prefix) :].strip() if prefix and text.startswith(prefix) else text.strip()


def _similarity(left: str, right: str, corpus: list[str]) -> float:
    docs = [_tokens(text) for text in corpus]
    df = Counter(token for doc in docs for token in set(doc))
    n = max(1, len(docs))

    def vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {
            token: (1.0 + math.log(freq)) * max(0.12, math.log((n + 1) / (df.get(token, 0) + 1)))
            for token, freq in counts.items()
        }

    a, b = vector(_tokens(left)), vector(_tokens(right))
    na = math.sqrt(sum(value * value for value in a.values()))
    nb = math.sqrt(sum(value * value for value in b.values()))
    if not na or not nb:
        return 0.0
    return sum(value * b.get(token, 0.0) for token, value in a.items()) / (na * nb)


def _short_answer(answer: str) -> str:
    answer = re.sub(r"\[[^\]]+\]", " ", str(answer))
    return " ".join(answer.split())[:160]


class ErrorCorrectionMemory(MemorySystem):
    """Retain only failed predictions, paired with the labels that corrected them."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.corrections: list[dict[str, str]] = []
        self._seen: set[str] = set()
        self.max_corrections = 200
        self.max_retrieved = 5

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        corpus = [item["input"] for item in self.corrections]
        prefix = _shared_prefix(corpus + [input]) if corpus else ""
        query = _body(input, prefix)
        ranked = sorted(
            [
                (
                _similarity(query, _body(item["input"], prefix), [_body(text, prefix) for text in corpus]),
                index,
                item,
                )
                for index, item in enumerate(self.corrections)
            ],
            key=lambda row: (row[0], row[1]),
            reverse=True,
        )
        selected = [row[2] for row in ranked[: self.max_retrieved] if row[0] > 0]
        correction_text = "\n\n".join(
            f"Earlier problem: {_body(item['input'], prefix)}\nEarlier wrong answer: {item['prediction']}\n"
            f"Correct label: {item['target']}"
            for item in selected
        )
        if not correction_text:
            correction_text = "No similar corrected cases have been stored yet."
        self._last_context_chars = len(correction_text)
        response = self.call_llm(PROMPT.format(corrections=correction_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_corrections": len(self.corrections),
            "num_retrieved": len(selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            if result.get("was_correct", False):
                continue
            item = {
                "input": result["input"],
                "prediction": _short_answer(result.get("prediction", "")),
                "target": str(result["ground_truth"]),
            }
            key = hashlib.sha256(
                json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if key not in self._seen:
                self._seen.add(key)
                self.corrections.append(item)
        self.corrections = self.corrections[-self.max_corrections :]
        self._seen = {
            hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            for item in self.corrections
        }

    def get_state(self) -> str:
        return json.dumps({"corrections": self.corrections}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.corrections = json.loads(state).get("corrections", [])[-self.max_corrections :]
        self._seen = {
            hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            for item in self.corrections
        }

    def get_context_length(self) -> int:
        return getattr(self, "_last_context_chars", 0)
