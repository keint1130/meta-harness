"""Represent labeled history as compact, per-class lexical cue profiles."""

import json
import math
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_COMMON_WORDS = {
    "the", "and", "for", "are", "but", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "has", "have", "this", "that",
    "with", "from", "they", "will", "would", "there", "their", "what", "when",
    "where", "which", "while", "into", "about", "been", "being", "were", "does",
    "did", "than", "then", "them", "some", "such", "only", "very", "also",
    "its", "his", "she", "him", "who", "how", "why", "because", "could",
    "should", "more", "most", "many", "much", "each", "other", "over", "under",
}
MAX_CUES_PER_LABEL = 7


def _profile_terms(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) > 2 and token not in _COMMON_WORDS
    }


class ClassCueProfiles(MemorySystem):
    """Store document frequencies, then surface terms that distinguish labels."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.total_documents = 0
        self.profiles: dict[str, dict[str, Any]] = {}

    def _add_document(self, text: str, label: str) -> None:
        profile = self.profiles.setdefault(label, {"documents": 0, "terms": {}})
        profile["documents"] += 1
        profile_terms = profile["terms"]
        for term in _profile_terms(text):
            profile_terms[term] = profile_terms.get(term, 0) + 1
        self.total_documents += 1

    def _cue_lines(self) -> str:
        labels = sorted(self.profiles)
        lines = []
        for label in labels:
            profile = self.profiles[label]
            n_positive = int(profile["documents"])
            n_negative = self.total_documents - n_positive
            terms = set(profile["terms"])
            for other_label in labels:
                if other_label != label:
                    terms.update(self.profiles[other_label]["terms"])

            scored = []
            for term in terms:
                positive = int(profile["terms"].get(term, 0))
                negative = sum(
                    int(other["terms"].get(term, 0))
                    for other_label, other in self.profiles.items()
                    if other_label != label
                )
                # Smoothed log odds reward terms frequent in this class and rarer elsewhere.
                score = math.log((positive + 0.5) / (n_positive - positive + 0.5))
                if n_negative:
                    score -= math.log((negative + 0.5) / (n_negative - negative + 0.5))
                if score > 0:
                    scored.append((score, term))
            scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
            cues = ", ".join(term for _, term in scored[:MAX_CUES_PER_LABEL])
            lines.append(f"- {label} ({n_positive} examples): {cues or 'no clear terms yet'}")
        return "\n".join(lines)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles = self._cue_lines() or "(No labeled examples yet.)"
        prompt = f"""Classify the input using the learned class profiles as compact evidence. The terms below are patterns from labeled examples, not required words; consider meaning and the complete input. Choose the appropriate label and return it exactly.

Class profiles:
{profiles}

Input:
{input}

Return only a JSON object with a concise string field named "final_answer" and an optional "reasoning" field."""
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_labels": len(self.profiles)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self._add_document(str(result["input"]), str(result["ground_truth"]))

    def get_context_length(self) -> int:
        return len(self._cue_lines())

    def get_state(self) -> str:
        return json.dumps(
            {"total_documents": self.total_documents, "profiles": self.profiles},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state or "{}")
        self.total_documents = int(data.get("total_documents", 0))
        self.profiles = data.get("profiles", {})
