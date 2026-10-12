"""Prompt for an explicit preserve/change analysis in one solution pass."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Solve the task below. First identify the decisive change or operation
that explains the requested result, and distinguish it from the information
that should remain unchanged. Use that distinction to check your answer, then
provide the requested deliverable. Follow every instruction in the task. Return
JSON with concise reasoning and a final_answer field containing only the
requested answer.

{input}
"""


class Candidate(MemorySystem):
    """Use a single-pass invariant and change analysis before answering."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response = self.call_llm(PROMPT.format(input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
