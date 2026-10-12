"""Aggregate nearest-case evidence into label-balanced kernel scores."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the evidence table as a set of retrieved examples, not as a fixed answer key.
Compare the current problem with the strongest examples for each plausible label. A label's
support score aggregates its closest cases, so inspect both the score and the details. Follow
the requested response format and return one exact label from the problem or examples.

Retrieved label evidence:
{evidence}

Problem:
{input}

Return JSON only: {{\"reasoning\": \"brief rationale\", \"final_answer\": \"exact label\"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "up", "was", "we",
    "were", "when", "which", "with", "you", "your", "please", "provide", "answer",
    "problem", "question", "label", "format", "final", "based", "following",
}


def _shared_edges(texts: list[str]) -> tuple[str, str]:
    if not texts:
        return "", ""
    prefix = texts[0]
    suffix = texts[0]
    for text in texts[1:]:
        n = min(len(prefix), len(text))
        i = 0
        while i < n and prefix[i] == text[i]:
            i += 1
        prefix = prefix[:i]
        n = min(len(suffix), len(text))
        i = 0
        while i < n and suffix[-i - 1] == text[-i - 1]:
            i += 1
        suffix = suffix[-i:] if i else ""
    return (prefix if len(prefix) >= 32 else "", suffix if len(suffix) >= 32 else "")


def _body(text: str, edges: tuple[str, str]) -> str:
    prefix, suffix = edges
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP]


def _cosine(a: Counter[str], b: Counter[str], idf: dict[str, float]) -> float:
    dot = sum(a[t] * b.get(t, 0) * idf.get(t, 0.0) ** 2 for t in a)
    na = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in a.items()))
    nb = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in b.items()))
    return dot / (na * nb) if na and nb else 0.0


class LabelBalancedKernel(MemorySystem):
    """Score labels from several close examples instead of showing a flat neighbor list."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_labels = 6
        self.examples_per_label = 2
        self._last_context_chars = 0

    def _evidence(self, query: str) -> str:
        if not self.examples:
            return "No labeled examples have been stored yet."
        raw = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _shared_edges(raw + [query])
        docs = [_tokens(_body(text, edges)) for text in raw]
        query_tokens = Counter(_tokens(_body(query, edges)))
        df = Counter(t for doc in docs for t in set(doc))
        idf = {t: max(0.15, math.log((len(docs) + 1) / (freq + 1))) for t, freq in df.items()}
        vectors = [Counter(doc) for doc in docs]
        by_label: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for index, (doc, ex) in enumerate(zip(vectors, self.examples)):
            by_label[ex["target"]].append((_cosine(query_tokens, doc, idf), index))

        ranked = []
        for label, neighbors in by_label.items():
            neighbors.sort(reverse=True)
            top = neighbors[: self.examples_per_label]
            # The best neighbor remains influential while another independent case can confirm it.
            support = 0.72 * top[0][0] + 0.28 * sum(s for s, _ in top[1:]) / max(1, len(top) - 1)
            ranked.append((support, label, top))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        lines = []
        for score, label, neighbors in ranked[: self.max_labels]:
            lines.append(f"Label: {label} | retrieved support: {score:.3f}")
            for similarity, index in neighbors:
                if similarity <= 0:
                    continue
                detail = _body(raw[index], edges).replace("\n", " ")[:220]
                lines.append(f"  Similarity {similarity:.3f}; example: {detail}")
        return "\n".join(lines) or "No examples share useful terms with this problem."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        evidence = self._evidence(input)
        self._last_context_chars = len(evidence)
        response = self.call_llm(PROMPT.format(evidence=evidence, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_labels_shown": min(self.max_labels, len({ex["target"] for ex in self.examples})),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            metadata = result.get("metadata") or {}
            raw = result.get("raw_question") or metadata.get("raw_question") or result["input"]
            self.examples.append({"input": str(result["input"]), "raw_question": str(raw), "target": str(result["ground_truth"])})

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])

    def get_context_length(self) -> int:
        return self._last_context_chars
