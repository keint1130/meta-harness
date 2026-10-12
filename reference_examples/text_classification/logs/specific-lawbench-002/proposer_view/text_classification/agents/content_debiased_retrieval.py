"""Retrieve complete labeled examples while discounting repeated vocabulary."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Use the relevant labeled examples as demonstrations. Repeated wording may be boilerplate; match the substantive content and follow the current problem's instructions.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer, preserving the answer form requested by the problem.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _features(text: str) -> list[str]:
    result: list[str] = []
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.extend(run[i : i + 2] for i in range(len(run) - 1))
            result.extend(run[i : i + 3] for i in range(len(run) - 2))
        else:
            result.append(run)
    return result


def _source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        row.get("raw_question")
        or metadata.get("raw_question")
        or metadata.get("question")
        or row.get("input", "")
    )


def _clip(text: str, limit: int = 1350) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


class Candidate(MemorySystem):
    """Query-conditioned BM25 retrieval with explicit boilerplate discounting."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 12500
        self.max_examples = 14
        self._last_context = ""

    def _rank(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        documents = [_features(example["input"]) for example in self.examples]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        count = len(documents)
        average_length = sum(map(len, documents)) / max(1, count)
        query_counts = Counter(_features(query))
        ranked: list[tuple[float, int]] = []
        for index, document in enumerate(documents):
            term_counts = Counter(document)
            score = 0.0
            for token, query_count in query_counts.items():
                frequency = document_frequency.get(token, 0)
                term_count = term_counts.get(token, 0)
                if not frequency or not term_count:
                    continue
                # Near-universal terms contribute little; the hard taper also
                # prevents repeated prompt boilerplate from dominating matches.
                prevalence = frequency / count
                if prevalence >= 0.92:
                    continue
                idf = math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
                idf *= max(0.08, 1.0 - prevalence**2)
                denominator = term_count + 1.2 * (
                    0.25 + 0.75 * len(document) / max(1.0, average_length)
                )
                score += min(query_count, 2) * idf * term_count * 2.2 / denominator
            ranked.append((score, index))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def _format_examples(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        ranked = self._rank(query)
        selected = [index for _, index in ranked[: self.max_examples]]
        for index in selected:
            example = self.examples[index]
            part = f"Q: {_clip(example['input'])}\nA: {example['target']}"
            remaining = self.context_budget - used
            if remaining < 100:
                break
            if len(part) > remaining:
                part = part[: remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No learned examples yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_examples": len(
                [part for part in self._last_context.split("\n\n") if part]
            ),
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(example["input"], example["target"]) for example in self.examples}
        for row in batch_results:
            item = {"input": _source_text(row), "target": str(row["ground_truth"])}
            key = (item["input"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._last_context = ""
