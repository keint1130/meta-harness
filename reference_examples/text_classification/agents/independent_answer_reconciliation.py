"""Use independent solution attempts, then reconcile them against the task."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


ATTEMPT_PROMPT = """Solve the task below independently. Follow all its
instructions and return a JSON object with "reasoning" and "final_answer".

{input}
"""

RECONCILE_PROMPT = """Two independent candidate responses were produced for the
task. Re-evaluate the original task and compare the candidates. Resolve any
disagreement from the task's evidence and constraints; if they agree, still
check that the shared answer follows the requested format. Do not assume either
candidate is correct. Return only JSON with "reasoning" and "final_answer".

Original task:
{input}

Candidate response A:
{first}

Candidate response B:
{second}
"""


class Candidate(MemorySystem):
    """Reconcile two hidden-from-each-other initial attempts."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        first = self.call_llm(ATTEMPT_PROMPT.format(input=input))
        second = self.call_llm(ATTEMPT_PROMPT.format(input=input))
        final = self.call_llm(
            RECONCILE_PROMPT.format(input=input, first=first, second=second)
        )
        return extract_json_field(final, "final_answer"), {
            "full_response": final,
            "attempt_a": extract_json_field(first, "final_answer"),
            "attempt_b": extract_json_field(second, "final_answer"),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
