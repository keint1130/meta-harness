"""A bounded ledger of recent prediction errors and their corrections."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem, using previous corrections as cautions where they apply.

{corrections_section}

**Problem:**
{input}

The correction records show earlier failures and their correct outputs. Compare
the reasoning pattern with this problem, but do not copy an output unless the
full problem supports it. Respond in JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}"""

MAX_STORED_ERRORS = 24
MAX_CONTEXT_ERRORS = 6


class CorrectionLedger(MemorySystem):
    """Keep only failed cases, prioritizing the most recent corrections."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.errors: list[dict[str, str]] = []

    def _format_corrections(self) -> str:
        recent = self.errors[-MAX_CONTEXT_ERRORS:]
        if not recent:
            return "No previous correction records are available."

        rows = []
        for index, error in enumerate(recent, start=1):
            rows.append(
                f"Correction {index}:\n"
                f"Earlier problem: {error['input']}\n"
                f"Earlier attempt: {error['prediction']}\n"
                f"Correct output: {error['target']}"
            )
        return "Recent corrected failures:\n\n" + "\n\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        corrections = self._format_corrections()
        prompt = PROMPT_TEMPLATE.format(corrections_section=corrections, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_stored_errors": len(self.errors),
            "num_context_errors": min(len(self.errors), MAX_CONTEXT_ERRORS),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            if result.get("was_correct", False):
                continue
            self.errors.append(
                {
                    "input": str(result.get("raw_question") or result["input"]),
                    "prediction": str(result["prediction"]),
                    "target": str(result["ground_truth"]),
                }
            )
        self.errors = self.errors[-MAX_STORED_ERRORS:]

    def get_state(self) -> str:
        return json.dumps({"errors": self.errors}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.errors = data.get("errors", [])

