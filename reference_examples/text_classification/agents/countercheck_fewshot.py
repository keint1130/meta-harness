"""Use a skeptical second pass to audit a few-shot draft against precedents."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


DRAFT_PROMPT = """Solve the problem using the labeled precedents.

{examples_section}

Problem:
{input}

Use the complete problem and follow its answer format. Return JSON with
reasoning and final_answer.
"""

AUDIT_PROMPT = """Audit this draft answer against the problem and labeled precedents.

Labeled precedents:
{examples_section}

Problem:
{input}

Draft answer: {draft_answer}
Draft reasoning: {draft_reasoning}

Act as a skeptical verifier. Identify the strongest precedent supporting the
draft and the strongest conflicting precedent. Check whether their relevant
facts actually match the full problem. Keep or revise the answer based on that
comparison. Follow the answer format requested in the problem. Return JSON
with reasoning and final_answer.
"""


def _seed_for_input(input: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(input.encode()).digest()[:8], "big")


class CountercheckFewshot(MemorySystem):
    """Draft with all learned examples, then independently audit the draft."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    @staticmethod
    def _example_text(example: dict[str, str]) -> str:
        return example.get("raw_question", example["input"])

    def _format_examples(self, input: str) -> str:
        if not self.examples:
            return "No labeled precedents have been learned yet."
        # A stable rotation varies presentation across problems without changing coverage.
        start = _seed_for_input(input) % len(self.examples)
        ordered = self.examples[start:] + self.examples[:start]
        return "\n\n".join(
            f"Precedent:\nProblem: {self._example_text(example)}\n"
            f"Correct output: {example['target']}"
            for example in ordered
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples(input)
        draft_response = self.call_llm(
            DRAFT_PROMPT.format(examples_section=examples_section, input=input)
        )
        draft_answer = extract_json_field(draft_response, "final_answer")
        draft_reasoning = extract_json_field(draft_response, "reasoning")[:1000]

        audit_response = self.call_llm(
            AUDIT_PROMPT.format(
                examples_section=examples_section,
                input=input,
                draft_answer=draft_answer,
                draft_reasoning=draft_reasoning,
            )
        )
        answer = extract_json_field(audit_response, "final_answer") or draft_answer
        return answer, {
            "full_response": audit_response,
            "draft_response": draft_response,
            "draft_answer": draft_answer,
            "num_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "input": str(result["input"]),
                    "raw_question": str(result.get("raw_question") or result["input"]),
                    "target": str(result["ground_truth"]),
                }
            )

    def get_context_length(self) -> int:
        return len(self._format_examples(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
