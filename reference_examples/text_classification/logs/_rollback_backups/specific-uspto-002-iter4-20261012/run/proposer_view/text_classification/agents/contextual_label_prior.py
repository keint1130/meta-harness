"""Use a high-confidence answer prior only when verified labels recur."""

import json
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """{prior}

Answer the following question, reasoning from its actual content:

{input}
"""


def _context(text: str) -> str:
    match = re.search(r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:", text)
    return re.sub(r"\s+", " ", match.group(1).strip().casefold()) if match else "__global__"


class Candidate(MemorySystem):
    """Store label frequencies instead of question-answer demonstrations."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.counts: dict[str, Counter[str]] = {"__all__": Counter()}
        self.seen: set[tuple[str, str, str]] = set()
        self.min_context_examples = 4
        self.min_context_share = 0.75
        self.min_global_examples = 5
        self.min_global_share = 0.90
        self.max_label_chars = 160

    def _prior_for(self, input: str) -> tuple[str, int, int] | None:
        context = _context(input)
        options = []
        if context != "__global__":
            options.append(
                (
                    self.counts.get(context, Counter()),
                    self.min_context_examples,
                    self.min_context_share,
                )
            )
        options.append(
            (
                self.counts.get("__all__", Counter()),
                self.min_global_examples,
                self.min_global_share,
            )
        )
        for counts, minimum, share_threshold in options:
            total = sum(counts.values())
            if total < minimum:
                continue
            label, frequency = max(counts.items(), key=lambda item: (item[1], item[0]))
            if len(label) <= self.max_label_chars and frequency / total >= share_threshold:
                return label, frequency, total
        return None

    def _format_prior(self, prior: tuple[str, int, int] | None) -> str:
        if prior is None:
            return ""
        label, frequency, total = prior
        return (
            "A compact prior from earlier verified examples in this context is "
            f"{label!r} ({frequency}/{total}). Treat it as a weak clue and verify "
            "it against the current problem."
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        prior = self._prior_for(input)
        prompt = PROMPT.format(prior=self._format_prior(prior), input=input)
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "used_label_prior": prior is not None,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            target = str(row.get("ground_truth", "")).strip()
            input_text = str(row.get("input", ""))
            context = _context(input_text)
            if not target:
                continue
            key = (context, input_text, target)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.counts.setdefault(context, Counter())[target] += 1
            self.counts.setdefault("__all__", Counter())[target] += 1

    def get_context_length(self) -> int:
        context_keys = [key for key in self.counts if key != "__all__"]
        priors = [self._format_prior(self._prior_for(f"Context: {key} Input:")) for key in context_keys]
        priors = [item for item in priors if item]
        if not priors:
            return 0
        return round(sum(map(len, priors)) / len(priors))

    def get_state(self) -> str:
        return json.dumps(
            {
                "counts": {key: dict(counter) for key, counter in self.counts.items()},
                "seen": [list(item) for item in sorted(self.seen)],
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.counts = {
            key: Counter(values) for key, values in data.get("counts", {}).items()
        }
        self.counts.setdefault("__all__", Counter())
        self.seen = {tuple(item) for item in data.get("seen", [])}
