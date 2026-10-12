"""Losslessly factor shared input scaffolding out of stored demonstrations."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using the compact demonstrations below.

{examples_section}

**Problem:**
{input}

The demonstrations share the displayed prefix and suffix. Each variable span
fits between them to reconstruct that example's full input. Compare the complete
reconstructed examples and their outputs; treat them as evidence, not rules.
Respond in JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}"""


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        limit = min(len(prefix), len(value))
        index = 0
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _common_suffix(values: list[str]) -> str:
    tails = [value[::-1] for value in values]
    return _common_prefix(tails)[::-1]


class ScaffoldMemory(MemorySystem):
    """Store mappings while emitting repeated input text only once."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    def _factor(self) -> tuple[str, str, list[str]]:
        inputs = [example["input"] for example in self.examples]
        prefix = _common_prefix(inputs)
        tails = [value[len(prefix):] for value in inputs]
        suffix = _common_suffix(tails)
        if suffix:
            spans = [tail[:-len(suffix)] for tail in tails]
        else:
            spans = tails
        return prefix, suffix, spans

    def _format_examples(self) -> str:
        if not self.examples:
            return "No demonstrations have been learned yet."

        prefix, suffix, spans = self._factor()
        rows = [
            "Shared input prefix (JSON string): "
            + json.dumps(prefix, ensure_ascii=False),
            "Shared input suffix (JSON string): "
            + json.dumps(suffix, ensure_ascii=False),
            "Mapping pairs [variable input span, correct output] as JSON:",
        ]
        pairs = [[span, example["target"]] for span, example in zip(spans, self.examples)]
        rows.append(json.dumps(pairs, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples()
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        prefix, suffix, _ = self._factor()
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "shared_prefix_chars": len(prefix),
            "shared_suffix_chars": len(suffix),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "input": str(result.get("raw_question") or result["input"]),
                    "target": str(result["ground_truth"]),
                }
            )

    def get_context_length(self) -> int:
        return len(self._format_examples())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
