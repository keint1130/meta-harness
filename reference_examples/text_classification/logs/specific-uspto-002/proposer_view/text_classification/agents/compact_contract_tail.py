"""Direct prompting with a short, task-agnostic final format check."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Perform the task described below. Read its constraints carefully.

{input}

Before responding, check that the final answer follows the required format and
contains only the requested result. Return a JSON object with "reasoning" and
"final_answer" fields; put the task's answer in "final_answer".
"""


class Candidate(MemorySystem):
    """Keep the original task intact and add one compact answer check at its end."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response = self.call_llm(PROMPT.format(input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "format_check_added": True,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
