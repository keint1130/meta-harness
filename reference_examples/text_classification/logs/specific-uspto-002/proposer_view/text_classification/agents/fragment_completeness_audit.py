"""Audit precursor completeness and fragment-level consistency."""
import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Solve the task below and follow the exact answer format.
Return a JSON object containing reasoning and final_answer.

{input}
"""

FRAGMENT_AUDIT_PROMPT = """Check the draft specifically for completeness of the
requested answer. Inventory the distinct product regions that must be accounted
for by the proposed precursor set. For each, decide whether it is retained,
formed, removed, or supplied by a reaction partner. Check that all essential
reactant fragments and any explicitly required counterions or coproducts are
included, and that no unrelated reagent has been added. Revisit ambiguous
small fragments rather than silently omitting them. Preserve exact task
formatting. Return JSON with reasoning and final_answer.

Original task:
{input}

Draft:
{draft}
"""


class Candidate(MemorySystem):
    """Use fragment accounting to catch missing or spurious precursors."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        draft = self.call_llm(DRAFT_PROMPT.format(input=input))
        review = self.call_llm(FRAGMENT_AUDIT_PROMPT.format(input=input, draft=draft))
        return extract_json_field(review, "final_answer"), {
            "full_response": review,
            "draft_response": draft,
            "draft_answer": extract_json_field(draft, "final_answer"),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        json.loads(state)
        self._state = state
