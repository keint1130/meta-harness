"""Extract a task-neutral constraint checklist before producing an answer."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PLAN_PROMPT = """Read the task and identify its required deliverable, hard
constraints, and response format. Do not solve it yet. Return a concise
checklist in plain text.

Task:
{input}
"""

SOLVE_PROMPT = """Complete the original task using the checklist as a final
coverage check. Treat the original task as authoritative if the checklist is
incomplete. Follow its requested response format exactly.

Original task:
{input}

Checklist:
{checklist}

Return a JSON object with "reasoning" and "final_answer" fields. Put only the
requested deliverable in "final_answer".
"""


class Candidate(MemorySystem):
    """Separate instruction parsing from task execution."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        checklist = self.call_llm(PLAN_PROMPT.format(input=input))
        response = self.call_llm(
            SOLVE_PROMPT.format(input=input, checklist=checklist.strip())
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "checklist": checklist,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
