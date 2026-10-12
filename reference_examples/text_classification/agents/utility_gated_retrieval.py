"""High-precision, context-matched retrieval with abstention."""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


BASE_PROMPT = """Answer the following question.

{input}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your reasoning]",
  "final_answer": "[Your concise final answer]"
}}
"""

MEMORY_PROMPT = """A single verified example with a closely matching problem and
the same declared context is available below. Use it only if its details apply;
solve the current problem independently.

{example}

Answer the following question and follow all instructions in the problem:

{input}

Return the requested answer in the format specified above.
"""


def _case_text(text: str) -> str:
    match = re.search(
        r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:\s*(.*?)(?:\bAnswer\s*:|$)",
        text,
    )
    if match:
        return "Context: " + match.group(1).strip() + " Input: " + match.group(2).strip()
    match = re.search(
        r"(?is)\b(?:Question|Problem|Query)\s*:\s*(.*?)(?:\bAnswer\s*:|$)",
        text,
    )
    return match.group(1).strip() if match else text.strip()


def _context(text: str) -> str:
    match = re.search(r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:", text)
    return re.sub(r"\s+", " ", match.group(1).strip().casefold()) if match else "__global__"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+|[^\w\s]", text.casefold()))


def _chargrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.casefold())
    return {compact[i : i + 3] for i in range(max(0, len(compact) - 2))}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


class Candidate(MemorySystem):
    """Retrieve one close same-context example, or abstain entirely."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        # Pooled traces put only a minority of same-context neighbors above
        # this conservative composite overlap, which limits noisy insertions.
        self.min_similarity = 0.48
        self.max_examples = 1

    def _rank(self, input: str) -> list[tuple[float, dict[str, str]]]:
        query = _case_text(input)
        query_context = _context(query)
        query_tokens, query_grams = _tokens(query), _chargrams(query)
        ranked = []
        for example in self.examples:
            if query_context != _context(example["case"]):
                continue
            score = 0.55 * _jaccard(query_tokens, _tokens(example["case"]))
            score += 0.45 * _jaccard(query_grams, _chargrams(example["case"]))
            ranked.append((score, example))
        ranked.sort(key=lambda item: (item[0], item[1]["case"]), reverse=True)
        return ranked

    def _selected(self, input: str) -> list[tuple[float, dict[str, str]]]:
        ranked = self._rank(input)
        if not ranked or ranked[0][0] < self.min_similarity:
            return []
        return ranked[: self.max_examples]

    def _format(self, selected: list[tuple[float, dict[str, str]]]) -> str:
        if not selected:
            return ""
        score, example = selected[0]
        return (
            f"Verified example (overlap {score:.2f}):\n{example['case']}\n"
            f"Verified answer: {example['target']}"
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._selected(input)
        prompt = (
            MEMORY_PROMPT.format(example=self._format(selected), input=input)
            if selected
            else BASE_PROMPT.format(input=input)
        )
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_examples": len(selected),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        known = {(item["case"], item["target"]) for item in self.examples}
        for row in batch_results:
            target = str(row.get("ground_truth", "")).strip()
            case = _case_text(str(row.get("input", "")))
            key = (case, target)
            if case and target and key not in known:
                self.examples.append({"case": case, "target": target})
                known.add(key)

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        lengths = [
            len(self._format(self._selected(example["case"])))
            for example in self.examples
        ]
        return round(sum(lengths) / len(lengths))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
