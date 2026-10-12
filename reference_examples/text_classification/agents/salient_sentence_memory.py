"""Store extractive, salience-ranked sentence summaries for labeled examples."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using these concise labeled case extracts.

{examples_section}

**Problem:**
{input}

The extracts retain selected sentences from earlier examples. They are partial
evidence, so compare all available details in the new problem before answering.
Respond in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

SENTENCES_PER_CASE = 4
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;.!?])|[\r\n]+")


def _char_ngrams(text: str) -> set[str]:
    features: set[str] = set()
    for chunk in _WORD_RE.findall(text.lower()):
        for size in (2, 3):
            if len(chunk) >= size:
                features.update(
                    chunk[index : index + size]
                    for index in range(len(chunk) - size + 1)
                )
    return features


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class SalientSentenceMemory(MemorySystem):
    """Keep high-information sentence extracts instead of full case text."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.summaries: list[str] = []

    @staticmethod
    def _text(example: dict[str, str]) -> str:
        return example.get("raw_question") or example["input"]

    def _refresh_summaries(self) -> None:
        sentence_groups = [_sentences(self._text(example)) for example in self.examples]
        sentence_features = [
            [_char_ngrams(sentence) for sentence in group]
            for group in sentence_groups
        ]
        document_frequency: Counter[str] = Counter()
        sentence_count = 0
        for group in sentence_features:
            for features in group:
                document_frequency.update(features)
                sentence_count += 1
        total_sentences = max(1, sentence_count)

        summaries = []
        for sentences, feature_group in zip(sentence_groups, sentence_features):
            scored = []
            for index, (sentence, features) in enumerate(zip(sentences, feature_group)):
                salience = sum(
                    math.log((total_sentences + 1) / (document_frequency[feature] + 1))
                    + 1.0
                    for feature in features
                ) / math.sqrt(max(1, len(sentence)))
                scored.append((salience, index, sentence))
            selected = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[
                :SENTENCES_PER_CASE
            ]
            selected.sort(key=lambda item: item[1])
            summaries.append(" ".join(sentence for _, _, sentence in selected))
        self.summaries = summaries

    def _format_examples(self) -> str:
        if not self.examples:
            return "No labeled examples have been learned yet."
        rows = []
        for index, (example, summary) in enumerate(
            zip(self.examples, self.summaries), start=1
        ):
            rows.append(
                f"Case {index}: {summary}\n"
                f"Correct output: {example['target']}"
            )
        return "Extracted labeled examples:\n\n" + "\n\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples()
        response = self.call_llm(
            PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_sentences_per_case": SENTENCES_PER_CASE,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            example = {
                "input": str(result["input"]),
                "target": str(result["ground_truth"]),
            }
            if "raw_question" in result:
                example["raw_question"] = str(result["raw_question"])
            self.examples.append(example)
        if batch_results:
            self._refresh_summaries()

    def get_context_length(self) -> int:
        return len(self._format_examples())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._refresh_summaries()
