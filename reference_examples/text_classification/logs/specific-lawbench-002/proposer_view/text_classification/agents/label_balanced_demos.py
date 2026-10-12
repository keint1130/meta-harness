"""Retrieve ordered, label-diverse demonstrations under a fixed context budget."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Solve the current problem using the demonstrations as examples of the requested task and answer format. Prefer the closest factual analogies. When several examples are relevant, compare their decisive facts; do not copy an answer solely because its wording overlaps.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer that follows the requested format.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _features(text: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            out.update(run[i:i + 2] for i in range(len(run) - 1))
            out.update(run[i:i + 3] for i in range(len(run) - 2))
        else:
            out[run] += 1
    return out


def _source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("raw_question") or metadata.get("raw_question") or metadata.get("question") or row.get("input", ""))


class Candidate(MemorySystem):
    """Use nearest examples first, then add output-label breadth as demonstrations."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 15000
        self.max_examples = 24
        self._last_context = ""

    def _rank(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        docs = [_features(row["input"]) for row in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = len(docs)
        idf = {t: math.log(1.0 + (n + 1) / (f + 1)) for t, f in df.items()}

        def vector(counts: Counter[str]) -> dict[str, float]:
            raw = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in counts.items() if c}
            norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        q = vector(_features(query))
        ranked = []
        for i, doc in enumerate(docs):
            d = vector(doc)
            ranked.append((sum(v * d.get(t, 0.0) for t, v in q.items()), i))
        return sorted(ranked, reverse=True)

    def _select(self, query: str) -> list[int]:
        ranked = self._rank(query)
        chosen: list[int] = []
        labels: set[str] = set()
        remaining = list(ranked)
        while remaining and len(chosen) < self.max_examples:
            best_pos, best_value = -1, -1.0
            for pos, (sim, idx) in enumerate(remaining):
                if sim <= 0:
                    continue
                label_bonus = 0.10 if self.examples[idx]["target"] not in labels else 0.0
                value = sim + label_bonus / (1 + len(chosen) * 0.08)
                if value > best_value:
                    best_pos, best_value = pos, value
            if best_pos < 0:
                break
            _, idx = remaining.pop(best_pos)
            chosen.append(idx)
            labels.add(self.examples[idx]["target"])
        return chosen

    def _format(self, query: str) -> str:
        parts: list[str] = []
        used = 0
        for idx in self._select(query):
            ex = self.examples[idx]
            part = f"Q: {ex['input']}\nA: {ex['target']}"
            remaining = self.context_budget - used
            if remaining < 120:
                break
            if len(part) > remaining:
                part = part[:remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No demonstrations have been learned yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._last_context.split("\n\n")) if self._last_context else 0}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["input"], x["target"]) for x in self.examples}
        for row in batch_results:
            ex = {"input": _source(row), "target": str(row["ground_truth"])}
            if (ex["input"], ex["target"]) not in seen:
                self.examples.append(ex)
                seen.add((ex["input"], ex["target"]))

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
