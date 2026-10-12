"""Retrieve verified examples together with explicit counterexamples from feedback."""

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the prior cases as calibration evidence. A verified label is the known correct
outcome. If a case is marked as incorrect, its recorded model answer is a counterexample and
must not be copied as a recommendation. Compare the current problem with both successful
examples and corrected mistakes, then follow the answer format requested in the problem.

Prior cases:
{cases}

Problem:
{input}

Return JSON only: {{"reasoning": "brief rationale", "final_answer": "answer"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "up", "was", "we",
    "were", "when", "which", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _shared_edges(texts: list[str]) -> tuple[str, str]:
    if not texts:
        return "", ""
    prefix, suffix = texts[0], texts[0]
    for text in texts[1:]:
        limit = min(len(prefix), len(text))
        index = 0
        while index < limit and prefix[index] == text[index]:
            index += 1
        prefix = prefix[:index]
        limit = min(len(suffix), len(text))
        index = 0
        while index < limit and suffix[-index - 1] == text[-index - 1]:
            index += 1
        suffix = suffix[-index:] if index else ""
    return (prefix if len(prefix) >= 40 else "", suffix if len(suffix) >= 40 else "")


def _body(text: str, edges: tuple[str, str]) -> str:
    prefix, suffix = edges
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.strip()


def _cosine(left: Counter[str], right: Counter[str], idf: dict[str, float]) -> float:
    dot = sum(left[token] * right.get(token, 0) * idf.get(token, 0.0) ** 2 for token in left)
    norm_left = math.sqrt(sum(value * value * idf.get(token, 0.0) ** 2 for token, value in left.items()))
    norm_right = math.sqrt(sum(value * value * idf.get(token, 0.0) ** 2 for token, value in right.items()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _short(text: Any, limit: int = 120) -> str:
    return " ".join(str(text).split())[:limit]


class OutcomeContrastMemory(MemorySystem):
    """Store both correct outcomes and explicit wrong-answer/correct-label pairs."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.max_examples = 500
        self.max_retrieved = 7
        self.max_chars = 4200
        self._seen: set[str] = set()
        self._last_context_chars = 0

    def _retrieve(self, input: str) -> tuple[list[dict[str, Any]], tuple[str, str]]:
        if not self.examples:
            return [], ("", "")
        raw = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _shared_edges(raw + [input])
        docs = [_tokens(_body(text, edges)) for text in raw]
        query = Counter(_tokens(_body(input, edges)))
        df = Counter(token for doc in docs for token in set(doc))
        count = max(1, len(docs))
        idf = {token: max(0.12, math.log((count + 1) / (freq + 1))) for token, freq in df.items()}
        query_vector = query
        ranked = []
        for index, (doc, example) in enumerate(zip(docs, self.examples)):
            score = _cosine(query_vector, Counter(doc), idf)
            ranked.append((score, index, example))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        # Reserve space for mistakes as well as successful examples so a memory
        # of errors does not crowd out positive category evidence (or vice versa).
        selected: list[dict[str, Any]] = []
        counts: Counter[bool] = Counter()
        quotas = {True: 4, False: 3}
        for score, _, example in ranked:
            if score <= 0 or counts[bool(example["was_correct"])] >= quotas[bool(example["was_correct"])]:
                continue
            selected.append(example)
            counts[bool(example["was_correct"])] += 1
            if len(selected) >= self.max_retrieved:
                break
        if not selected:
            selected = self.examples[-min(3, len(self.examples)) :]
        return selected, edges

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected, edges = self._retrieve(input)
        parts = []
        for example in selected:
            details = _body(example.get("raw_question", example["input"]), edges).replace("\n", " ")
            if example["was_correct"]:
                part = f"Problem details: {details}\nVerified label: {example['target']} (prior answer was correct)"
            else:
                part = (
                    f"Problem details: {details}\nIncorrect prior answer: {example['prediction']}\n"
                    f"Verified label: {example['target']}"
                )
            parts.append(part)
        case_text = "\n\n".join(parts) or "No prior feedback cases have been stored yet."
        if len(case_text) > self.max_chars:
            case_text = case_text[: self.max_chars]
        self._last_context_chars = len(case_text)
        response = self.call_llm(PROMPT.format(cases=case_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_retrieved": len(selected),
            "num_error_cases": sum(not bool(ex["was_correct"]) for ex in selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            metadata = result.get("metadata") or {}
            raw = result.get("raw_question") or metadata.get("raw_question") or result["input"]
            item = {
                "input": str(result["input"]),
                "raw_question": str(raw),
                "prediction": _short(result.get("prediction", "")),
                "target": str(result["ground_truth"]),
                "was_correct": bool(result.get("was_correct", False)),
            }
            key = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            if key not in self._seen:
                self._seen.add(key)
                self.examples.append(item)
        self.examples = self.examples[-self.max_examples :]
        self._seen = {
            hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            for item in self.examples
        }

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])[-self.max_examples :]
        self._seen = {
            hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            for item in self.examples
        }

    def get_context_length(self) -> int:
        return self._last_context_chars
