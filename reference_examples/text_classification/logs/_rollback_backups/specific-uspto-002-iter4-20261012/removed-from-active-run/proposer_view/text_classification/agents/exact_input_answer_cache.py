"""Reuse a verified answer only for an exactly repeated normalized input."""

import json
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Answer the following task, following all of its instructions.

{input}

Return a JSON object with "reasoning" and "final_answer" fields. The
"final_answer" value must contain the requested answer in its required format.
"""


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class Candidate(MemorySystem):
    """Use verified labels only when the complete input has appeared before."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.answers: dict[str, Counter[str]] = {}

    def _unambiguous_answer(self, input: str) -> str | None:
        counts = self.answers.get(_key(input), Counter())
        if len(counts) != 1:
            return None
        return next(iter(counts))

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response = self.call_llm(PROMPT.format(input=input))
        generated = extract_json_field(response, "final_answer")
        cached = self._unambiguous_answer(input)
        return (cached if cached is not None else generated), {
            "full_response": response,
            "exact_cache_hit": cached is not None,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            key = _key(str(row.get("input", "")))
            target = str(row.get("ground_truth", "")).strip()
            if key and target:
                self.answers.setdefault(key, Counter())[target] += 1

    def get_context_length(self) -> int:
        # Exact answers are returned directly and are not added to the prompt.
        return 0

    def get_state(self) -> str:
        return json.dumps(
            {key: dict(counts) for key, counts in self.answers.items()},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.answers = {
            key: Counter(counts) for key, counts in data.items()
        }
