"""Retrieve one anchor per label and expose query-conditioned contrasts."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Compare the retrieved anchors as competing explanations of the problem. Each anchor
was selected because its wording overlaps the current problem; overlap is evidence, not proof.
Pay attention to which specific query details distinguish one label from its closest alternative.
Use the task's requested output format and return one exact label.

Query-conditioned contrasts:
{contrasts}

Problem:
{input}

Return JSON only: {{\"reasoning\": \"brief rationale\", \"final_answer\": \"exact label\"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "they", "this", "to", "up", "was", "we", "were", "when", "which", "with", "you", "your",
    "please", "provide", "answer", "problem", "question", "label", "format", "final", "based",
    "following", "symptoms", "example", "examples",
}


def _edges(texts: list[str]) -> tuple[str, str]:
    if not texts:
        return "", ""
    prefix = suffix = texts[0]
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


def _features(text: str) -> Counter[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]
    features = Counter(words)
    for left, right in zip(words, words[1:]):
        features[left + "_" + right] += 1.5
    return features


def _cosine(a: Counter[str], b: Counter[str], idf: dict[str, float]) -> float:
    dot = sum(a[t] * b.get(t, 0.0) * idf.get(t, 0.0) ** 2 for t in a)
    na = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in a.items()))
    nb = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in b.items()))
    return dot / (na * nb) if na and nb else 0.0


class ContrastiveAnchorMemory(MemorySystem):
    """Build query-specific comparisons from the closest case for each candidate label."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_anchors = 5
        self._last_context_chars = 0

    def _contrast_text(self, query: str) -> str:
        if not self.examples:
            return "No labeled anchors have been stored yet."
        raw = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _edges(raw + [query])
        docs = [_features(_body(text, edges)) for text in raw]
        q = _features(_body(query, edges))
        df = Counter(t for doc in docs for t in set(doc))
        idf = {t: max(0.15, math.log((len(docs) + 1) / (freq + 1))) for t, freq in df.items()}
        best_by_label: dict[str, tuple[float, int]] = {}
        for index, (doc, ex) in enumerate(zip(docs, self.examples)):
            label = ex["target"]
            score = _cosine(q, doc, idf)
            if label not in best_by_label or score > best_by_label[label][0]:
                best_by_label[label] = (score, index)
        ranked = sorted(((score, label, index) for label, (score, index) in best_by_label.items()), reverse=True)
        chosen = ranked[: self.max_anchors]
        query_terms = set(q)

        def matches(index: int) -> list[str]:
            overlap = query_terms & set(docs[index])
            return [term.replace("_", " ") for term in sorted(overlap, key=lambda t: (-(idf.get(t, 0.0) * q[t]), t))[:6]]

        lines = []
        for rank, (score, label, index) in enumerate(chosen):
            anchor_terms = set(docs[index]) & query_terms
            peer_rows = [row for row in chosen if row[1] != label]
            nearest_peer = peer_rows[0] if peer_rows else None
            contrast = []
            if nearest_peer:
                peer_terms = set(docs[nearest_peer[2]]) & query_terms
                contrast = list(anchor_terms - peer_terms) + list(peer_terms - anchor_terms)
                contrast.sort(key=lambda t: (-idf.get(t, 0.0) * q[t], t))
            lines.append(f"Anchor {rank + 1}: label={label}; retrieval relevance={score:.3f}")
            lines.append(f"  Query details matched: {', '.join(matches(index)) or 'no distinctive exact term'}")
            if nearest_peer and contrast:
                lines.append(f"  Terms separating this anchor from {nearest_peer[1]}: {', '.join(t.replace('_', ' ') for t in contrast[:6])}")
            detail = _body(raw[index], edges).replace("\n", " ")[:240]
            lines.append(f"  Labeled example: {detail}")
        return "\n".join(lines) or "No useful lexical overlap was found."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        contrasts = self._contrast_text(input)
        self._last_context_chars = len(contrasts)
        response = self.call_llm(PROMPT.format(contrasts=contrasts, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_anchors": min(self.max_anchors, len({ex["target"] for ex in self.examples})),
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
