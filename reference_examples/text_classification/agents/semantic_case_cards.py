"""Replace each labeled example with an LLM-written semantic case card."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


CARD_PROMPT = """Compress this labeled example into one compact case card.

Capture the concrete evidence in the input that supports the correct output.
Preserve constraints or exceptions that could change the answer. Avoid
unsupported generalizations and incidental identifiers. Keep the card under
80 words. Return JSON: {{"case_card": "..."}}

Correct output: {target}
Input: {input}
"""

PREDICT_PROMPT = """Solve the problem using the labeled semantic case cards.

{cards_section}

Cards preserve decision-relevant evidence from earlier examples. Treat each
card as partial evidence, compare the full input, and follow the answer format
requested in the problem. Return JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}

Problem:
{input}
"""


class SemanticCaseCards(MemorySystem):
    """Store one concise semantic record for every labeled training case."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.cards: list[dict[str, str]] = []

    def _make_card(self, text: str, target: str) -> str:
        response = self.call_llm(CARD_PROMPT.format(target=target, input=text))
        card = extract_json_field(response, "case_card").strip()
        return (card or text[:800])[:1000]

    def _format_cards(self) -> str:
        if not self.cards:
            return "No labeled case cards have been learned yet."
        rows = ["Labeled semantic precedents:"]
        for index, item in enumerate(self.cards, start=1):
            rows.append(
                f"Case {index}: {item['card']}\n"
                f"Correct output: {item['target']}"
            )
        return "\n\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        cards_section = self._format_cards()
        response = self.call_llm(
            PREDICT_PROMPT.format(cards_section=cards_section, input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_cards": len(self.cards),
            "card_chars": len(cards_section),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            text = str(result.get("raw_question") or result["input"])
            target = str(result["ground_truth"])
            self.cards.append({"card": self._make_card(text, target), "target": target})

    def get_context_length(self) -> int:
        return len(self._format_cards())

    def get_state(self) -> str:
        return json.dumps({"cards": self.cards}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.cards = json.loads(state).get("cards", [])
