"""Check whether proposed precursors can plausibly regenerate the product."""
import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Solve the task below, following every instruction and format rule.
Return a JSON object with reasoning and final_answer fields.

{input}
"""

ROUNDTRIP_PROMPT = """Audit the proposed answer by imagining the forward reaction.
Starting only from the listed precursor structures and the stated reaction
context, check whether the product in the original task can be regenerated with
its key bonds, functional groups, substituents, and stereochemistry intact.
Identify any mismatch such as an incorrect leaving group, oxidation state, or
unaccounted transformation. If the proposal fails this check, solve the task
again from the product and correct it. Return JSON with reasoning and
final_answer; final_answer must contain only the requested precursor answer.

Original task:
{input}

Proposed answer:
{answer}
"""


class Candidate(MemorySystem):
    """Use a forward-direction plausibility check to validate retrosynthesis."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        draft = self.call_llm(DRAFT_PROMPT.format(input=input))
        answer = extract_json_field(draft, "final_answer")
        check = self.call_llm(ROUNDTRIP_PROMPT.format(input=input, answer=answer))
        return extract_json_field(check, "final_answer"), {
            "full_response": check,
            "draft_answer": answer,
            "draft_response": draft,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        json.loads(state)
        self._state = state
