"""Aggregate independent answers by exact normalized vote."""

import json
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


ATTEMPT_PROMPT = """Solve the task independently, following all its instructions.
Return a JSON object with reasoning and final_answer fields.

{input}
"""

RESOLVE_PROMPT = """Choose the best supported answer from these independent
candidates by checking each against the original task. Treat them as proposals,
not evidence, and do not invent a new format. Return JSON with reasoning and
final_answer fields.

Original task:
{input}

Candidate answers:
{answers}
"""


def _key(value: str) -> str:
    return " ".join(value.split())


class Candidate(MemorySystem):
    """Take an exact-answer majority when available; otherwise resolve options."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        responses = [self.call_llm(ATTEMPT_PROMPT.format(input=input)) for _ in range(3)]
        answers = [extract_json_field(r, "final_answer").strip() for r in responses]
        counts = Counter(_key(a) for a in answers if a)
        winner, count = counts.most_common(1)[0] if counts else ("", 0)
        if count >= 2:
            chosen = next(a for a in answers if _key(a) == winner)
            return chosen, {"candidate_answers": answers, "vote_count": count}
        options = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers))
        resolved = self.call_llm(RESOLVE_PROMPT.format(input=input, answers=options))
        return extract_json_field(resolved, "final_answer"), {
            "full_response": resolved,
            "candidate_answers": answers,
            "vote_count": count,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
