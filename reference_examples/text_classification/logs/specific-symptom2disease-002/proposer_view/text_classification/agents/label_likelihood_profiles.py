"""Represent memory as smoothed, label-specific lexical evidence."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """The profiles below summarize words that distinguish learned labels. Their lexical
scores are clues, not guaranteed answers. Compare the problem's strongest details with the
profiles and the representative examples, and consider close alternative labels carefully.
Follow the answer format requested in the problem and use one exact learned label when possible.

Label evidence:
{profiles}

Problem:
{input}

Return JSON only: {{"reasoning": "brief rationale", "final_answer": "answer"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "there", "they", "this", "to", "up", "was", "we", "were",
    "when", "which", "with", "you", "your", "also", "always", "days", "feel",
    "feeling", "getting", "having", "lately", "lot", "lots", "much", "now",
    "noticed", "really", "sometimes",
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


class LabelLikelihoodProfiles(MemorySystem):
    """Learn smoothed token evidence per label, with one compact example each."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_labels = 8
        self._last_context_chars = 0

    def _profile_lines(self, input: str) -> list[str]:
        if not self.examples:
            return []
        raw = [ex.get("raw_question", ex["input"]) for ex in self.examples]
        edges = _shared_edges(raw + [input])
        docs = [_tokens(_body(text, edges)) for text in raw]
        label_docs: dict[str, list[int]] = defaultdict(list)
        for index, ex in enumerate(self.examples):
            label_docs[ex["target"]].append(index)

        df = Counter(token for doc in docs for token in set(doc))
        query = Counter(_tokens(_body(input, edges)))
        count = max(1, len(docs))
        label_scores: dict[str, float] = {}
        label_terms: dict[str, list[tuple[float, str]]] = {}
        label_matches: dict[str, list[str]] = {}
        representative: dict[str, str] = {}

        for label, indexes in label_docs.items():
            in_count = Counter(token for index in indexes for token in set(docs[index]))
            index_set = set(indexes)
            outside_count = Counter(
                token
                for index, doc in enumerate(docs)
                if index not in index_set
                for token in set(doc)
            )
            inside_n = len(indexes)
            outside_n = count - inside_n
            odds: dict[str, float] = {}
            for token in df:
                # A token seen in only one example is too fragile to define a
                # category profile; the representative example carries that
                # information without turning a stray word into a strong cue.
                if in_count[token] < 2:
                    continue
                p_in = (in_count[token] + 0.5) / (inside_n + 1.0)
                p_out = (outside_count[token] + 0.5) / (outside_n + 1.0)
                odds[token] = math.log(p_in / p_out)
            # A mild prior resolves ties without making frequent labels dominate.
            score = math.log((inside_n + 0.5) / (count + 0.5 * len(label_docs)))
            score += sum(min(2, freq) * odds.get(token, 0.0) for token, freq in query.items())
            label_scores[label] = score
            label_terms[label] = sorted(
                ((weight, token) for token, weight in odds.items() if weight > 0.05),
                reverse=True,
            )[:6]
            matches = sorted(
                ((odds.get(token, 0.0), token) for token in query), reverse=True
            )
            label_matches[label] = [token for weight, token in matches if weight > 0][:3]

            # Pick the example most representative of its same-label peers.
            best_index = max(
                indexes,
                key=lambda index: sum(
                    len(set(docs[index]) & set(docs[other]))
                    / max(1, len(set(docs[index]) | set(docs[other])))
                    for other in indexes
                    if other != index
                ) / max(1, len(indexes) - 1),
            )
            representative[label] = _body(raw[best_index], edges).replace("\n", " ").strip()[:180]

        ordered = sorted(label_docs, key=lambda label: (label_scores[label], label), reverse=True)
        lines = []
        for label in ordered[: self.max_labels]:
            cues = label_matches[label] or [token for _, token in label_terms[label][:3]]
            cue_text = ", ".join(cues) if cues else "no distinctive token overlap"
            lines.append(
                f"Label: {label} | lexical fit: {label_scores[label]:+.2f} | cues: {cue_text}\n"
                f"Profile terms: {', '.join(token for _, token in label_terms[label][:5]) or '(few examples)'}\n"
                f"Representative problem: {representative[label]}"
            )
        return lines

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles = self._profile_lines(input)
        profile_text = "\n\n".join(profiles) or "No label evidence has been learned yet."
        self._last_context_chars = len(profile_text)
        response = self.call_llm(PROMPT.format(profiles=profile_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_labels_shown": len(profiles),
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
