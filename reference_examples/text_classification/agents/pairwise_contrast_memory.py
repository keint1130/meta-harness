"""Present query-matched, label-pair contrast cards learned from examples."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using these learned contrasts between labels.

{contrast_section}

**Problem:**
{input}

The contrast cards summarize words that distinguished the labeled examples in
each pair. Treat them as weak clues, compare the full problem, and select the
best-supported answer. Respond in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

MAX_CARDS = 8
MAX_CUES_PER_SIDE = 3


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{2,}", text.lower()))


class PairwiseContrastMemory(MemorySystem):
    """Learn discriminative terms for pairs of competing labels."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.class_token_docs: dict[str, Counter[str]] = {}
        self.class_docs: Counter[str] = Counter()

    def _cues(
        self, own: Counter[str], other: Counter[str], own_docs: int, other_docs: int
    ) -> list[str]:
        ranked = []
        for token, own_count in own.items():
            # Smoothed log-odds on document presence favors terms that separate
            # the two labels instead of terms that are frequent everywhere.
            other_count = other[token]
            score = math.log((own_count + 0.5) / (own_docs - own_count + 0.5))
            score -= math.log(
                (other_count + 0.5) / (other_docs - other_count + 0.5)
            )
            if score > 0:
                ranked.append((score, token))
        return [token for _, token in sorted(ranked, reverse=True)[:MAX_CUES_PER_SIDE]]

    def _cards(self) -> list[dict[str, Any]]:
        labels = sorted(self.class_token_docs, key=str.casefold)
        cards = []
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                cards.append(
                    {
                        "left": left,
                        "right": right,
                        "left_cues": self._cues(
                            self.class_token_docs[left],
                            self.class_token_docs[right],
                            self.class_docs[left],
                            self.class_docs[right],
                        ),
                        "right_cues": self._cues(
                            self.class_token_docs[right],
                            self.class_token_docs[left],
                            self.class_docs[right],
                            self.class_docs[left],
                        ),
                    }
                )
        return cards

    def _format_contrasts(self, query: str) -> str:
        query_tokens = _tokens(query)
        ranked = []
        for card in self._cards():
            left_hits = query_tokens.intersection(card["left_cues"])
            right_hits = query_tokens.intersection(card["right_cues"])
            hits = len(left_hits) + len(right_hits)
            if hits or not query:
                ranked.append((hits, max(len(left_hits), len(right_hits)), card))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                len(item[2]["left_cues"]) + len(item[2]["right_cues"]),
            ),
            reverse=True,
        )
        if not ranked:
            return "No learned contrast cues match this problem."

        lines = ["Query-matched label contrasts (left-side cues / right-side cues):"]
        for _, _, card in ranked[:MAX_CARDS]:
            left = ", ".join(card["left_cues"]) or "no cue"
            right = ", ".join(card["right_cues"]) or "no cue"
            lines.append(f"- {card['left']} versus {card['right']}: {left} / {right}")
        return "\n".join(lines)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        contrast_section = self._format_contrasts(input)
        prompt = PROMPT_TEMPLATE.format(contrast_section=contrast_section, input=input)
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_label_pairs": len(self._cards()),
            "contrast_chars": len(contrast_section),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            label = str(result["ground_truth"])
            text = str(result.get("raw_question") or result["input"])
            self.class_docs[label] += 1
            self.class_token_docs.setdefault(label, Counter()).update(_tokens(text))

    def get_context_length(self) -> int:
        return len(self._format_contrasts(""))

    def get_state(self) -> str:
        return json.dumps(
            {
                "class_token_docs": {
                    label: dict(counts)
                    for label, counts in self.class_token_docs.items()
                },
                "class_docs": dict(self.class_docs),
            },
            ensure_ascii=False,
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.class_token_docs = {
            label: Counter(counts)
            for label, counts in data.get("class_token_docs", {}).items()
        }
        self.class_docs = Counter(data.get("class_docs", {}))
