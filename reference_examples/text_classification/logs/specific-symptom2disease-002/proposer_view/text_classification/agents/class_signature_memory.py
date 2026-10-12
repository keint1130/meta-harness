"""Represent learned labels with compact discriminative terms and medoid examples."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Choose the label that best fits the problem. The profiles are compact summaries
of labeled examples; use the example wording and distinctive features to compare categories.
Use the exact spelling of one learned label, follow any answer format explicitly requested
in the problem, and do not add qualifications.

Learned label profiles:
{profiles}

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


def _trim_shared_prefix(texts: list[str]) -> str:
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        limit = min(len(prefix), len(text))
        pos = 0
        while pos < limit and prefix[pos] == text[pos]:
            pos += 1
        prefix = prefix[:pos]
    return prefix if len(prefix) >= 40 else ""


def _body(text: str, prefix: str) -> str:
    return text[len(prefix) :].strip() if prefix and text.startswith(prefix) else text.strip()


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1 and token not in _STOP]


def _cosine(left: list[str], right: list[str], idf: dict[str, float]) -> float:
    a, b = Counter(left), Counter(right)
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] * idf.get(t, 1.0) ** 2 for t in common)
    na = math.sqrt(sum(v * v * idf.get(t, 1.0) ** 2 for t, v in a.items()))
    nb = math.sqrt(sum(v * v * idf.get(t, 1.0) ** 2 for t, v in b.items()))
    return dot / (na * nb) if na and nb else 0.0


class ClassSignatureMemory(MemorySystem):
    """Compress examples into label signatures and one representative instance each."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    def _profiles(self) -> list[str]:
        if not self.examples:
            return []
        inputs = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        prefix = _trim_shared_prefix(inputs)
        docs = [_tokens(_body(text, prefix)) for text in inputs]
        labels: dict[str, list[int]] = defaultdict(list)
        for index, ex in enumerate(self.examples):
            labels[ex["target"]].append(index)

        df = Counter(token for doc in docs for token in set(doc))
        total = len(docs)
        idf = {token: math.log((total + 1) / (freq + 1)) + 0.2 for token, freq in df.items()}
        global_counts = Counter(token for doc in docs for token in set(doc))
        profiles = []
        for label in sorted(labels):
            indexes = labels[label]
            in_count = Counter(token for idx in indexes for token in set(docs[idx]))
            outside = max(0, total - len(indexes))
            in_size = len(indexes)
            terms = []
            for token, count in in_count.items():
                other_count = global_counts[token] - count
                p_in = (count + 0.5) / (in_size + 1.0)
                p_out = (other_count + 0.5) / (outside + 1.0)
                contrast = math.log(p_in / (1 - p_in)) - math.log(p_out / (1 - p_out))
                terms.append((contrast * idf.get(token, 1.0), token))
            signature = ", ".join(token for _, token in sorted(terms, reverse=True)[:5])

            medoid_idx = max(
                indexes,
                key=lambda idx: sum(
                    _cosine(docs[idx], docs[other], idf)
                    for other in indexes
                    if other != idx
                ) / max(1, len(indexes) - 1),
            )
            question = self.examples[medoid_idx].get("raw_question", self.examples[medoid_idx]["input"])
            question = _body(question, prefix).replace("\n", " ").strip()[:260]
            profiles.append(
                f"Label: {label}\nDistinctive terms: {signature or '(none yet)'}\n"
                f"Representative example: {question}"
            )
        return profiles

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles = self._profiles()
        response = self.call_llm(PROMPT.format(profiles="\n\n".join(profiles), input=input))
        return extract_json_field(response, "final_answer"), {
            "num_labels": len(profiles),
            "num_examples": len(self.examples),
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
        return len("\n\n".join(self._profiles()))
