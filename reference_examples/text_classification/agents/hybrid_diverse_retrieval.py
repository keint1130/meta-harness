"""Retrieve demonstrations with complementary lexical signals and diversity."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use these labeled examples to identify the category that best matches the problem.
Compare the specific details, give more weight to distinctive matches than to generic overlap,
and consider why a close example with a different label may differ. Follow the requested
answer format in the problem and use the exact wording of a learned label when appropriate.

{examples}

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
    # Keep negation terms: they can reverse the meaning of an otherwise close match.
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _shared_edges(texts: list[str]) -> tuple[str, str]:
    if not texts:
        return "", ""
    prefix = texts[0]
    suffix = texts[0]
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
    common = left.keys() & right.keys()
    dot = sum(left[t] * right[t] * idf.get(t, 0.0) ** 2 for t in common)
    norm_left = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in left.items()))
    norm_right = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in right.items()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class HybridDiverseRetrieval(MemorySystem):
    """Blend TF-IDF cosine and BM25, then avoid redundant same-label neighbors."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 10
        self.max_per_label = 2
        self._last_context_chars = 0

    def _retrieve(self, input: str) -> tuple[list[dict[str, str]], tuple[str, str]]:
        if not self.examples:
            return [], ("", "")
        raw = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _shared_edges(raw + [input])
        docs = [_tokens(_body(text, edges)) for text in raw]
        query = Counter(_tokens(_body(input, edges)))
        doc_freq = Counter(token for doc in docs for token in set(doc))
        count = max(1, len(docs))
        idf = {
            token: max(0.12, math.log((count + 1) / (freq + 1)))
            for token, freq in doc_freq.items()
        }
        query_vector = query
        doc_vectors = [Counter(doc) for doc in docs]
        cosine_scores = [_cosine(query_vector, vector, idf) for vector in doc_vectors]

        average_length = max(1.0, sum(map(len, docs)) / count)
        bm25_scores = []
        for doc in docs:
            tf = Counter(doc)
            score = 0.0
            for token, qfreq in query.items():
                frequency = tf.get(token, 0)
                if not frequency:
                    continue
                bm25_idf = math.log(1.0 + (count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                denominator = frequency + 1.2 * (0.25 + 0.75 * len(doc) / average_length)
                score += min(2, qfreq) * bm25_idf * (frequency * 2.2 / denominator)
            bm25_scores.append(score)
        bm25_max = max(bm25_scores, default=0.0)
        candidates = []
        for index, example in enumerate(self.examples):
            bm25 = bm25_scores[index] / bm25_max if bm25_max else 0.0
            relevance = 0.55 * cosine_scores[index] + 0.45 * bm25
            candidates.append((relevance, index, example, doc_vectors[index]))

        # Greedy maximal marginal relevance keeps relevant cases while reducing
        # duplicate wording; the per-label cap prevents one frequent category
        # from filling the whole context.
        chosen: list[tuple[float, int, dict[str, str], Counter[str]]] = []
        label_counts: Counter[str] = Counter()
        remaining = list(candidates)
        while remaining and len(chosen) < self.max_examples:
            best_pos = None
            best_utility = float("-inf")
            for pos, candidate in enumerate(remaining):
                relevance, _, example, vector = candidate
                if label_counts[example["target"]] >= self.max_per_label:
                    continue
                redundancy = max(
                    (_cosine(vector, prior[3], idf) for prior in chosen),
                    default=0.0,
                )
                utility = relevance - 0.18 * redundancy
                if utility > best_utility:
                    best_pos, best_utility = pos, utility
            if best_pos is None:
                break
            candidate = remaining.pop(best_pos)
            chosen.append(candidate)
            label_counts[candidate[2]["target"]] += 1
        selected = [item[2] for item in chosen if item[0] > 0]
        if not selected:
            selected = self.examples[-min(3, len(self.examples)) :]
        return selected, edges

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected, edges = self._retrieve(input)
        example_text = "\n\n".join(
            f"Example problem: {_body(ex.get('raw_question', ex['input']), edges)}\n"
            f"Label: {ex['target']}"
            for ex in selected
        ) or "No labeled examples have been stored yet."
        self._last_context_chars = len(example_text)
        response = self.call_llm(PROMPT.format(examples=example_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_retrieved": len(selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            metadata = result.get("metadata") or {}
            raw = result.get("raw_question") or metadata.get("raw_question") or result["input"]
            self.examples.append(
                {"input": str(result["input"]), "raw_question": str(raw), "target": str(result["ground_truth"])}
            )

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])

    def get_context_length(self) -> int:
        return self._last_context_chars
