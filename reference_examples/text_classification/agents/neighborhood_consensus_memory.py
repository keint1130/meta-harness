"""Retrieve a local neighborhood and expose weighted target agreement."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PROMPT = """The memory gives a small neighborhood of labeled examples and a summary
of how their expected answers agree or differ. Similarity is evidence, not proof:
compare the full current problem before choosing. Follow the current problem's
instructions and required output format.

{summary}

{examples}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. The
final_answer must follow the current problem's requested format.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _terms(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _WORD_RE.findall(text))


def _vector(terms: Counter[str], df: Counter[str], n: int) -> dict[str, float]:
    weighted = {
        key: (1.0 + math.log(count)) * (1.0 + math.log((n + 1.0) / (df[key] + 1.0)))
        for key, count in terms.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    return {key: value / norm for key, value in weighted.items()}


class NeighborhoodConsensusMemory(MemorySystem):
    """Summarize answer support only among the query's closest stored cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _rank(self, query: str) -> list[tuple[float, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_terms(item["question"]) for item in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        query_vector = _vector(_terms(query), df, len(docs))
        ranked = []
        for item, doc in zip(self.examples, docs):
            vector = _vector(doc, df, len(docs))
            score = sum(value * vector.get(key, 0.0) for key, value in query_vector.items())
            ranked.append((score, item))
        return sorted(ranked, key=lambda pair: -pair[0])[:6]

    def _render(self, query: str) -> tuple[str, str]:
        neighbors = self._rank(query)
        support: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for score, item in neighbors:
            support[item["target"]] += score
            counts[item["target"]] += 1
        summary = "No nearby labeled examples yet."
        if support:
            order = sorted(support, key=lambda target: (-support[target], target))
            summary = "Local answer support (weighted by lexical similarity):\n" + "\n".join(
                f"- {target!r}: support {support[target]:.3f} across {counts[target]} nearby example(s)"
                for target in order[:5]
            )
        examples = "\n\n".join(
            f"Neighbor {rank} (similarity {score:.3f})\nProblem: {item['question']}\nExpected answer: {item['target']}"
            for rank, (score, item) in enumerate(neighbors, 1)
        ) or "No labeled examples yet."
        return summary, examples

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        summary, examples = self._render(input)
        self._last_context_chars = len(summary) + len(examples)
        response = self.call_llm(_PROMPT.format(summary=summary, examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_neighbors": examples.count("Neighbor "),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(item["question"], item["target"]) for item in self.examples}
        for result in batch_results:
            item = {"question": _question(result), "target": str(result.get("ground_truth", ""))}
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-512:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._render("")[0]) + len(self._render("")[1])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ][-512:]
        self._last_context_chars = 0
