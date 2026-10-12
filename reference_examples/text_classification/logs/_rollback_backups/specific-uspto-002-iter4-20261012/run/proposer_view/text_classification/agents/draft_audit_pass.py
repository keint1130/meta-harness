"""Use an independent second pass to audit a draft against the original task."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Solve the task below, following its instructions and required
answer format. Return a JSON object with "reasoning" and "final_answer" fields.

{input}
"""

AUDIT_PROMPT = """Review the draft against the original task. Check that it answers
the actual input, obeys all stated constraints, includes everything required,
and adds nothing the requested answer excludes. Correct any issue you find.
Return only a JSON object with "reasoning" and "final_answer" fields. The
"final_answer" value must use the exact format required by the task.

Original task:
{input}

Draft response:
{draft}
"""


class Candidate(MemorySystem):
    """Generate a draft, then ask for a focused consistency and format audit."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        draft = self.call_llm(DRAFT_PROMPT.format(input=input))
        draft_answer = extract_json_field(draft, "final_answer")
        final_response = self.call_llm(
            AUDIT_PROMPT.format(input=input, draft=draft)
        )
        return extract_json_field(final_response, "final_answer"), {
            "full_response": final_response,
            "draft_answer": draft_answer,
            "draft_response": draft,
            "audit_pass": True,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
