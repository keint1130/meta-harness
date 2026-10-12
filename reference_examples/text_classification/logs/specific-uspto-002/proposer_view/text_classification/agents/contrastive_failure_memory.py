"""Memory of prior failed attempts paired with their verified corrections."""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the prior failure records as warnings and verified corrections.
Do not repeat a failed approach when its underlying issue applies. Adapt the
correction to this problem and follow the requested answer format.

{failures}

Problem:
{input}

Return a JSON object with a concise reasoning field and a final_answer field.
"""


def _case_text(text: str) -> str:
    context = re.search(
        r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:\s*(.*?)(?:\bAnswer\s*:|$)",
        text,
    )
    if context:
        return "Context: " + context.group(1).strip() + " Input: " + context.group(2).strip()
    fields = re.search(r"(?is)\b(?:Question|Problem|Query)\s*:\s*(.*?)(?:\bAnswer\s*:|$)", text)
    return fields.group(1).strip() if fields else text.strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+|[^\w\s]", text.casefold()))


def _similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / len(left | right) if left and right else 0.0


class Candidate(MemorySystem):
    """Keep only erroneous predictions, alongside the correction that fixed each."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.failures: list[dict[str, str]] = []
        self.max_failures = 2

    def _rank(self, query: str) -> list[dict[str, str]]:
        case = _case_text(query)
        return sorted(
            self.failures,
            key=lambda ex: (_similarity(case, ex["case"]), ex["case"]),
            reverse=True,
        )[: self.max_failures]

    def _format(self, selected: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"Prior failed case:\n{ex['case']}\nFailed answer: {ex['attempt']}\n"
            f"Verified correction: {ex['target']}"
            for ex in selected
        ) or "No prior failed cases are available."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank(input)
        response = self.call_llm(PROMPT.format(failures=self._format(selected), input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "failure_examples": len(selected)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        known = {(ex["case"], ex["target"]) for ex in self.failures}
        for row in batch_results:
            target = str(row.get("ground_truth", "")).strip()
            attempt = str(row.get("prediction", "")).strip()
            was_correct = row.get("was_correct")
            if was_correct is None:
                was_correct = attempt == target
            case = _case_text(str(row.get("input", "")))
            key = (case, target)
            if not was_correct and target and case and key not in known:
                self.failures.append({"case": case, "attempt": attempt, "target": target})
                known.add(key)

    def get_context_length(self) -> int:
        if not self.failures:
            return 0
        lengths = [len(self._format(self._rank(ex["case"]))) for ex in self.failures]
        return round(sum(lengths) / len(lengths))

    def get_state(self) -> str:
        return json.dumps({"failures": self.failures}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.failures = data.get("failures", [])
