"""Retrieve query-relevant labeled examples with multilingual character BM25."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using the most relevant labeled precedents below.

{examples_section}

**Problem:**
{input}

Compare the complete facts with the precedents. Shared wording can suggest a
similarity, but use the full problem to choose the best supported output.
Respond in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

TOP_K = 5
BM25_K1 = 1.2
BM25_B = 0.75
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _char_ngrams(text: str) -> Counter[str]:
    features: Counter[str] = Counter()
    for chunk in _WORD_RE.findall(text.lower()):
        for size in (2, 3, 4):
            if len(chunk) >= size:
                features.update(
                    chunk[index : index + size]
                    for index in range(len(chunk) - size + 1)
                )
    return features


class CharNgramCaseRetrieval(MemorySystem):
    """Rank stored examples with BM25 over character n-grams."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    @staticmethod
    def _text(example: dict[str, str]) -> str:
        return example.get("raw_question") or example["input"]

    def _rank_examples(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        documents = [_char_ngrams(self._text(example)) for example in self.examples]
        query_terms = _char_ngrams(query)
        if not query_terms:
            start = max(0, len(self.examples) - TOP_K)
            return [(0.0, index) for index in range(start, len(self.examples))]

        document_frequency: Counter[str] = Counter()
        lengths = []
        for document in documents:
            document_frequency.update(document.keys())
            lengths.append(sum(document.values()))
        average_length = sum(lengths) / max(1, len(lengths)) or 1.0
        total = len(documents)

        ranked = []
        for index, document in enumerate(documents):
            length = lengths[index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (total - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / average_length
                )
                score += (
                    inverse_frequency
                    * query_frequency
                    * frequency
                    * (BM25_K1 + 1.0)
                    / denominator
                )
            ranked.append((score, index))
        ranked.sort(reverse=True)
        return ranked[:TOP_K]

    def _format_examples(self, query: str) -> str:
        selected = self._rank_examples(query)
        if not selected:
            return "No labeled precedents have been learned yet."
        rows = []
        for rank, (_, index) in enumerate(selected, start=1):
            example = self.examples[index]
            rows.append(
                f"Precedent {rank}:\n"
                f"Problem: {self._text(example)}\n"
                f"Correct output: {example['target']}"
            )
        return "Most textually relevant labeled precedents:\n\n" + "\n\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank_examples(input)
        examples_section = self._format_examples(input)
        response = self.call_llm(
            PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(selected),
            "top_bm25_score": selected[0][0] if selected else 0.0,
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

    def get_context_length(self) -> int:
        return len(self._format_examples(""))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
