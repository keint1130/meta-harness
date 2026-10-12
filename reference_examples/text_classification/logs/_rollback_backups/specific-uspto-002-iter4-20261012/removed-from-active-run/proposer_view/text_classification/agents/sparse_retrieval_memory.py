"""Compact case memory with lexical retrieval over the variable prompt region."""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Solve the problem using the most relevant verified examples below.
Transfer only the parts that apply to the current input. Preserve the requested
answer format.

{examples}

Problem:
{input}

Return a JSON object with a concise reasoning field and a final_answer field.
"""


def _case_text(text: str) -> str:
    """Keep prompt fields that describe this case, omitting repeated directions."""
    context = re.search(
        r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:\s*(.*?)(?:\bAnswer\s*:|$)",
        text,
    )
    if context:
        return "Context: " + context.group(1).strip() + " Input: " + context.group(2).strip()
    fields = re.search(r"(?is)\b(?:Question|Problem|Query)\s*:\s*(.*?)(?:\bAnswer\s*:|$)", text)
    if fields:
        return fields.group(1).strip()
    return text.strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+|[^\w\s]", text.casefold()))


def _chargrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.casefold())
    return {compact[i : i + 3] for i in range(max(0, len(compact) - 2))}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class Candidate(MemorySystem):
    """Retrieve a few prompt-near cases rather than append the entire history."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 4

    def _rank(self, query: str) -> list[dict[str, str]]:
        q = _case_text(query)
        q_tokens, q_grams = _tokens(q), _chargrams(q)
        ranked = []
        for ex in self.examples:
            e_tokens, e_grams = _tokens(ex["case"]), _chargrams(ex["case"])
            score = 0.55 * _jaccard(q_tokens, e_tokens) + 0.45 * _jaccard(q_grams, e_grams)
            ranked.append((score, ex))
        ranked.sort(key=lambda item: (item[0], item[1]["case"]), reverse=True)
        return [ex for _, ex in ranked[: self.max_examples]]

    def _format(self, selected: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"Verified example {i}:\n{ex['case']}\nVerified answer: {ex['target']}"
            for i, ex in enumerate(selected, 1)
        ) or "No prior examples are available."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank(input)
        response = self.call_llm(PROMPT.format(examples=self._format(selected), input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "retrieved_examples": len(selected)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        known = {ex["case"] for ex in self.examples}
        for row in batch_results:
            target = str(row.get("ground_truth", "")).strip()
            if not target:
                continue
            case = _case_text(str(row.get("input", "")))
            if case and case not in known:
                self.examples.append({"case": case, "target": target})
                known.add(case)

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        lengths = [len(self._format(self._rank(ex["case"]))) for ex in self.examples]
        return round(sum(lengths) / len(lengths))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
