"""Decompose a requested transformation into a reaction-center change before solving."""
import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


ANALYZE_PROMPT = """Analyze the product and task instructions before proposing reactants.
Return JSON with fields: reaction_summary, changed_bonds_or_groups,
unchanged_scaffold, candidate_precursor_changes, ambiguities. Use concise text.
Do not answer by copying the product. Treat the declared reaction context as a
hint and the actual product structure as the evidence.

Task:
{input}
"""

SOLVE_PROMPT = """Solve the original task and obey its exact output contract.
Use the transformation notes to focus on the changed region. Preserve the
unchanged scaffold and stereochemistry. Include every required precursor
fragment, and return only the requested answer in final_answer.

Original task:
{input}

Transformation notes:
{notes}
"""


class Candidate(MemorySystem):
    """Plan around a declared reaction center, then produce the requested answer."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        notes = self.call_llm(ANALYZE_PROMPT.format(input=input))
        response = self.call_llm(SOLVE_PROMPT.format(input=input, notes=notes))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "transformation_notes": notes,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        json.loads(state)
        self._state = state
