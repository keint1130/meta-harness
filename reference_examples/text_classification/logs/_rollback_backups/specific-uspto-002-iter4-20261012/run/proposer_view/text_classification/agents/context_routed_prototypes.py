"""Compact diverse prototypes grouped by an explicit context field."""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use these representative verified examples for the matching context.
They summarize varied cases, so transfer only a pattern that applies to the
current input. Follow the requested answer format.

{examples}

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


def _route(case: str) -> str:
    match = re.search(r"(?im)^\s*Context\s*:\s*(.*?)\s*(?:Input\s*:|$)", case)
    return re.sub(r"\s+", " ", match.group(1).strip().casefold()) if match else "__global__"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+|[^\w\s]", text.casefold()))


def _similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / len(left | right) if left and right else 0.0


class Candidate(MemorySystem):
    """Route by a declared context label and retain only diverse prototypes."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.prototypes: dict[str, list[dict[str, str]]] = {}
        self.max_per_route = 3

    def _add_prototype(self, route: str, example: dict[str, str]) -> None:
        bucket = self.prototypes.setdefault(route, [])
        if any(ex["case"] == example["case"] for ex in bucket):
            return
        if len(bucket) < self.max_per_route:
            bucket.append(example)
            return

        candidate_novelty = 1.0 - max(_similarity(example["case"], ex["case"]) for ex in bucket)
        redundancies = []
        for i, ex in enumerate(bucket):
            peers = bucket[:i] + bucket[i + 1 :]
            mean_similarity = sum(_similarity(ex["case"], peer["case"]) for peer in peers) / max(1, len(peers))
            redundancies.append((mean_similarity, i))
        most_redundant, index = max(redundancies)
        if candidate_novelty > 1.0 - most_redundant:
            bucket[index] = example

    def _selected(self, input: str) -> list[dict[str, str]]:
        case = _case_text(input)
        route = _route(case)
        selected = self.prototypes.get(route)
        if not selected:
            selected = self.prototypes.get("__global__", [])
        return selected[: self.max_per_route]

    def _format(self, selected: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"Representative example {i}:\n{ex['case']}\nVerified answer: {ex['target']}"
            for i, ex in enumerate(selected, 1)
        ) or "No matching prototypes are available."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._selected(input)
        response = self.call_llm(PROMPT.format(examples=self._format(selected), input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "prototype_count": len(selected)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            target = str(row.get("ground_truth", "")).strip()
            case = _case_text(str(row.get("input", "")))
            if target and case:
                self._add_prototype(_route(case), {"case": case, "target": target})

    def get_context_length(self) -> int:
        if not self.prototypes:
            return 0
        total, count = 0, 0
        for route, examples in self.prototypes.items():
            length = len(self._format(examples))
            total += length * len(examples)
            count += len(examples)
        return round(total / max(1, count))

    def get_state(self) -> str:
        return json.dumps({"prototypes": self.prototypes}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.prototypes = data.get("prototypes", {})
