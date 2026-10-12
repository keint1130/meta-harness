"""Select a compact set of both relevant and outcome-diverse examples."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The examples below were selected for relevance and coverage. Their answers are verified examples, but cases with different facts may have different answers. Use the closest evidence, then follow the current problem's instructions.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _features(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.update(run[i : i + 2] for i in range(len(run) - 1))
            result.update(run[i : i + 3] for i in range(len(run) - 2))
        else:
            result[run] += 1
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


def _clip(text: str, limit: int = 1050) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


class Candidate(MemorySystem):
    """Greedily select nearest examples while covering labels and query terms."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 8500
        self.max_examples = 14
        self._last_context = ""

    def _ranked_vectors(self, query: str) -> tuple[list[tuple[float, int]], list[set[str]]]:
        if not self.examples:
            return [], []
        documents = [_features(example["input"]) for example in self.examples]
        df: Counter[str] = Counter()
        for document in documents:
            df.update(document.keys())
        n_docs = len(documents)
        idf = {
            token: math.log(1.0 + (n_docs + 1) / (frequency + 1))
            for token, frequency in df.items()
        }

        def vector(counts: Counter[str]) -> dict[str, float]:
            raw = {token: (1.0 + math.log(count)) * idf.get(token, 0.0)
                   for token, count in counts.items() if count > 0}
            norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
            return {token: value / norm for token, value in raw.items()}

        doc_vectors = [vector(document) for document in documents]
        query_vector = vector(_features(query))
        query_terms = set(query_vector)
        scores = [
            sum(value * doc_vectors[index].get(token, 0.0)
                for token, value in query_vector.items())
            for index in range(n_docs)
        ]
        ranked = sorted(((score, index) for index, score in enumerate(scores)), reverse=True)
        return ranked, [set(document) & query_terms for document in documents]

    def _format_examples(self, query: str) -> str:
        ranked, overlaps = self._ranked_vectors(query)
        if not ranked:
            self._last_context = ""
            return "No learned examples yet."

        query_terms = set(_features(query))
        selected: list[int] = []
        selected_labels: set[str] = set()
        covered_terms: set[str] = set()
        parts: list[str] = []
        used = 0
        remaining_ranked = list(ranked)

        while remaining_ranked and len(selected) < self.max_examples:
            best_position = -1
            best_value = -1.0
            for position, (similarity, index) in enumerate(remaining_ranked):
                if similarity <= 0:
                    continue
                label = self.examples[index]["target"]
                unseen = overlaps[index] - covered_terms
                term_gain = len(unseen) / max(1, len(query_terms))
                value = 0.80 * similarity + (0.075 if label not in selected_labels else 0.0)
                value += 0.10 * min(1.0, term_gain * 7.0)
                if value > best_value:
                    best_position, best_value = position, value
            if best_position < 0:
                break

            _, index = remaining_ranked.pop(best_position)
            example = self.examples[index]
            label = example["target"]
            part = f"Related example:\nProblem: {_clip(example['input'])}\nVerified answer: {label}"
            if used + len(part) + 2 > self.context_budget:
                available = self.context_budget - used - len("Related example:\nProblem: \nVerified answer: ") - len(label)
                if available < 100:
                    continue
                part = f"Related example:\nProblem: {_clip(example['input'], available)}\nVerified answer: {label}"
                if used + len(part) + 2 > self.context_budget:
                    continue
            selected.append(index)
            selected_labels.add(label)
            covered_terms.update(overlaps[index])
            used += len(part) + 2
            # Save the rendered block beside its selection order for stable output.
            parts.append(part)

        self._last_context = "\n\n".join(parts) if selected else ""
        return self._last_context or "No overlapping learned examples yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_examples": len(self._last_context.split("\n\n")) if self._last_context else 0,
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(example["input"], example["target"]) for example in self.examples}
        for row in batch_results:
            example = {"input": _source_text(row), "target": str(row["ground_truth"])}
            key = (example["input"], example["target"])
            if key not in seen:
                self.examples.append(example)
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
