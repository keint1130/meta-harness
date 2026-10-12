"""Generate independent answers, then adjudicate them against the original task."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Solve the task below from scratch. Follow its instructions and
return a JSON object with concise reasoning and a final_answer field.

{input}
"""

ADJUDICATE_PROMPT = """Independently inspect the original task and the two proposed
answers. Check each answer against the task's actual evidence, requirements, and
output format. If they differ, resolve the disagreement from the task itself; if
they agree, still verify the result. Return only JSON with reasoning and the
best final_answer in exactly the requested format.

Original task:
{input}

Proposed answer A: {answer_a}
Proposed answer B: {answer_b}
"""


class Candidate(MemorySystem):
    """Use answer diversity as a signal for a final input-grounded decision."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response_a = self.call_llm(DRAFT_PROMPT.format(input=input))
        answer_a = extract_json_field(response_a, "final_answer")
        response_b = self.call_llm(DRAFT_PROMPT.format(input=input))
        answer_b = extract_json_field(response_b, "final_answer")
        final_response = self.call_llm(
            ADJUDICATE_PROMPT.format(
                input=input,
                answer_a=answer_a,
                answer_b=answer_b,
            )
        )
        return extract_json_field(final_response, "final_answer"), {
            "full_response": final_response,
            "candidate_answers": [answer_a, answer_b],
            "answers_agreed": answer_a == answer_b,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"

    def set_state(self, state: str) -> None:
        json.loads(state)
