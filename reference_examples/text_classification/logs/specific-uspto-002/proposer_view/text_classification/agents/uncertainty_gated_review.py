"""Spend a second pass only when a first-pass response flags uncertainty."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Answer the task below and follow its instructions and required
format. Return a JSON object with reasoning and final_answer fields.

{input}
"""

GATE_PROMPT = """Inspect the original task and the draft. Decide whether the draft
has a material chance of being wrong, incomplete, or format-invalid. Return only
JSON with a boolean field "needs_review". Be conservative: set it true when
you see a concrete uncertainty or possible violation.

Original task:
{input}

Draft:
{draft}
"""

REVIEW_PROMPT = """Re-solve the original task, paying particular attention to the
uncertainty identified in the draft. Follow the original instructions and return
JSON with reasoning and final_answer fields.

Original task:
{input}

Draft:
{draft}
"""


class Candidate(MemorySystem):
    """Use a compact uncertainty decision to allocate review effort."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        draft = self.call_llm(DRAFT_PROMPT.format(input=input))
        gate = self.call_llm(GATE_PROMPT.format(input=input, draft=draft))
        flag = extract_json_field(gate, "needs_review", "false").strip().lower()
        reviewed = flag in {"true", "1", "yes"}
        final = (self.call_llm(REVIEW_PROMPT.format(input=input, draft=draft))
                 if reviewed else draft)
        return extract_json_field(final, "final_answer"), {
            "full_response": final,
            "draft_response": draft,
            "needs_review": reviewed,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
