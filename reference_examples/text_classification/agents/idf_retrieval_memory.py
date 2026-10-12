"""Retrieve a compact, diverse set of labeled examples by lexical similarity."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the labeled examples as demonstrations of the task's categories.
Prefer the examples whose problem details are most similar to the current problem.
Return one concise answer using the exact label wording from the examples or input.
Follow any answer format explicitly requested in the problem, and do not add qualifications.

{examples}

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


def _common_edges(texts: list[str]) -> tuple[str, str]:
    """Find long literal wrappers shared by inputs so they do not dominate retrieval."""
    if not texts:
        return "", ""
    prefix = texts[0]
    for text in texts[1:]:
        limit = min(len(prefix), len(text))
        idx = 0
        while idx < limit and prefix[idx] == text[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            break
    suffix = texts[0]
    for text in texts[1:]:
        limit = min(len(suffix), len(text))
        idx = 0
        while idx < limit and suffix[-idx - 1] == text[-idx - 1]:
            idx += 1
        suffix = suffix[len(suffix) - idx :] if idx else ""
        if not suffix:
            break
    return (prefix if len(prefix) >= 40 else "", suffix if len(suffix) >= 40 else "")


def _body(text: str, edges: tuple[str, str]) -> str:
    prefix, suffix = edges
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _vectorize(tokens: list[str], df: Counter[str], count: int) -> dict[str, float]:
    tf = Counter(tokens)
    result = {}
    for token, frequency in tf.items():
        # A small floor keeps common words usable while strongly favoring rarer details.
        idf = max(0.12, math.log((count + 1) / (df.get(token, 0) + 1)))
        result[token] = (1.0 + math.log(frequency)) * idf
    return result


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    return dot / (norm_a * norm_b)


class IDFRetrievalMemory(MemorySystem):
    """Keep a small number of high-overlap demonstrations with label diversity."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 10
        self.max_per_label = 2
        self._last_context_chars = 0

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        corpus = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _common_edges(corpus + [input]) if corpus else ("", "")
        query = _body(input, edges)
        docs = [_tokens(_body(doc, edges)) for doc in corpus]
        df = Counter(token for doc in docs for token in set(doc))
        count = max(1, len(docs))
        doc_vectors = [_vectorize(doc, df, count) for doc in docs]
        query_vector = _vectorize(_tokens(query), df, count)
        ranked = []
        for index, ex in enumerate(self.examples):
            score = _cosine(query_vector, doc_vectors[index])
            ranked.append((score, index, ex))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected = []
        label_counts: Counter[str] = Counter()
        for score, _, ex in ranked:
            if score <= 0:
                continue
            label = ex["target"]
            if label_counts[label] >= self.max_per_label:
                continue
            selected.append(ex)
            label_counts[label] += 1
            if len(selected) >= self.max_examples:
                break
        if not selected and self.examples:
            selected = self.examples[-min(3, len(self.examples)) :]

        example_text = "\n\n".join(
            f"Example problem: {_body(ex.get('raw_question', ex['input']), edges)}\nLabel: {ex['target']}"
            for ex in selected
        )
        self._last_context_chars = len(example_text)
        response = self.call_llm(PROMPT.format(examples=example_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_retrieved": len(selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "input": result["input"],
                    "raw_question": result.get("raw_question", result["input"]),
                    "target": str(result["ground_truth"]),
                }
            )

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])

    def get_context_length(self) -> int:
        return getattr(self, "_last_context_chars", 0)
