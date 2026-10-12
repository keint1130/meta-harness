"""Retrieve verified examples using rare-feature coverage instead of raw overlap."""

import json
import math
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the verified references as limited evidence. Their order reflects
overlap on relatively rare text features; shared boilerplate contributes less.
Transfer a pattern only when the relevant details match the current task.

{examples}

Current task:
{input}

Return a JSON object with concise reasoning and a final_answer in the requested format.
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


def _features(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", text.casefold())
    words = {"w:" + token for token in re.findall(r"\w+", compact) if len(token) > 1}
    grams = {
        "g:" + compact[index : index + 4]
        for index in range(max(0, len(compact) - 3))
        if not compact[index : index + 4].isspace()
    }
    return words | grams


class Candidate(MemorySystem):
    """Weight matching features by corpus rarity and retrieve a small reference set."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 3

    def _rank(self, query: str) -> list[tuple[float, dict[str, str]]]:
        if not self.examples:
            return []
        query_features = _features(_case_text(query))
        if not query_features:
            return []

        document_features = [_features(example["case"]) for example in self.examples]
        frequency: dict[str, int] = {}
        for feature_set in document_features:
            for feature in feature_set:
                frequency[feature] = frequency.get(feature, 0) + 1

        n_docs = len(self.examples)
        weights = {
            feature: math.log(1.0 + (n_docs + 0.5) / (frequency.get(feature, 0) + 0.5))
            for feature in query_features
        }
        denominator = sum(weight * weight for weight in weights.values()) or 1.0
        ranked = []
        for example, features in zip(self.examples, document_features):
            shared = query_features & features
            score = sum(weights[feature] ** 2 for feature in shared) / denominator
            if score > 0:
                ranked.append((score, example))
        ranked.sort(key=lambda item: (item[0], item[1]["case"]), reverse=True)
        return ranked[: self.max_examples]

    def _format(self, ranked: list[tuple[float, dict[str, str]]]) -> str:
        if not ranked:
            return "No verified reference examples are available."
        return "\n\n".join(
            f"Reference {index} (rare-feature coverage {score:.2f}):\n"
            f"{example['case']}\nVerified answer: {example['target']}"
            for index, (score, example) in enumerate(ranked, 1)
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        ranked = self._rank(input)
        response = self.call_llm(
            PROMPT.format(examples=self._format(ranked), input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_examples": len(ranked),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        known = {(example["case"], example["target"]) for example in self.examples}
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
        lengths = [len(self._format(self._rank(example["case"]))) for example in self.examples]
        return round(sum(lengths) / len(lengths))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
