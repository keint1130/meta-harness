"""Label-grouped casebook prompt with explicit hypothesis comparison."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem by comparing the label hypotheses in the casebook.

{examples_section}

**Problem:**
{input}

**Comparison process:**
1. Identify the most informative facts in the problem.
2. Compare those facts with the cases under each plausible label.
3. For the strongest candidate, state its best supporting fact and the
   strongest conflicting fact. Eliminate candidates with weaker overall fit.
4. Use the complete pattern; a single shared detail is not enough.

Respond in JSON format:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000


class HypothesisGrouped(MemorySystem):
    """Store labeled cases and present them in competing label groups."""

    def __init__(
        self,
        llm: LLMCallable,
        max_examples: int = 9999,
    ):
        super().__init__(llm)
        self.max_examples = max_examples
        self.examples: list[dict[str, str]] = []

    def _format_examples_section(self) -> str:
        """Group every case under its label and obey the shared context cap."""
        if not self.examples:
            return ""
        groups: dict[str, list[dict[str, str]]] = {}
        for example in self.examples[-self.max_examples :]:
            groups.setdefault(example["target"], []).append(example)

        parts = []
        total_chars = 0
        for label in sorted(groups, key=str.casefold):
            cases = groups[label]
            header = f"LABEL: {label} ({len(cases)})"
            case_parts = [header]
            for example in cases:
                question = example.get("raw_question", example["input"])
                case_parts.append(f"- {question}")
            part = "\n".join(case_parts)
            if total_chars + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        """Generate prediction using accumulated few-shot examples."""
        examples_section = self._format_examples_section()
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_label_hypotheses": len({example["target"] for example in self.examples}),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Accumulate all examples with ground truth labels."""
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

    def get_context_length(self) -> int:
        """Return length of the examples section actually injected per query."""
        return len(self._format_examples_section())

    def get_state(self) -> str:
        """Serialize memory state."""
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        """Restore memory state from JSON."""
        data = json.loads(state)
        self.examples = data.get("examples", [])
