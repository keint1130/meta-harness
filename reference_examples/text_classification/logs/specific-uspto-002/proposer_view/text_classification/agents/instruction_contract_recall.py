"""Recall and emphasize a response contract repeated in task instructions."""

import json
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Answer the following question and follow the complete original task
instructions.

{input}

Repeated response requirement from the task instructions:
{contract}
"""


def _extract_contract(text: str) -> str:
    markers = list(
        re.finditer(
            r"(?i)\b(?:you must respond|respond (?:in|with)|answer in|"
            r"output (?:in|as|format)|return (?:a|an|your))\b",
            text,
        )
    )
    if not markers:
        return ""
    contract = text[markers[-1].start() :].strip()
    return contract[-1600:] if len(contract) > 1600 else contract


class Candidate(MemorySystem):
    """Move a repeatedly observed format contract to the prompt's final position."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.contract_counts: Counter[str] = Counter()
        self.contract = ""

    def _usable_contract(self, input: str) -> str:
        current = _extract_contract(input)
        if self.contract and current == self.contract:
            return self.contract
        # Before learning, or when the current prompt has its own variant,
        # repeat only the contract found in this input.
        return current

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        contract = self._usable_contract(input)
        prompt = (
            PROMPT.format(input=input, contract=contract)
            if contract
            else f"Answer the following question.\n\n{input}"
        )
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "recalled_contract": bool(contract and contract == self.contract),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            contract = _extract_contract(str(row.get("input", "")))
            if contract:
                self.contract_counts[contract] += 1
        eligible = [
            (count, contract)
            for contract, count in self.contract_counts.items()
            if count >= 2
        ]
        if eligible:
            self.contract = max(eligible, key=lambda item: (item[0], item[1]))[1]

    def get_context_length(self) -> int:
        return len(self.contract)

    def get_state(self) -> str:
        return json.dumps(
            {"contract_counts": dict(self.contract_counts), "contract": self.contract},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.contract_counts = Counter(data.get("contract_counts", {}))
        self.contract = data.get("contract", "")
