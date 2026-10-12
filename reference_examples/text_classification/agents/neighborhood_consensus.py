"""Pair local example retrieval with a similarity-weighted answer consensus."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The nearby cases provide both individual analogies and an approximate consensus over retrieved answers. Treat the consensus as a check, not a substitute for the current facts. Prefer a supported complete answer, including all distinct outcomes required by the current problem.

Similarity-weighted answer support:
{support}

Closest verified cases:
{examples}

Current problem:
{input}

Return JSON with a concise final_answer in the requested form.
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
    """Expose a weighted neighborhood answer distribution alongside examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.neighbor_count = 18
        self.example_count = 6
        self.context_budget = 10500
        self._last_context = ""

    def _neighbors(self, query: str) -> list[tuple[float, dict[str, str]]]:
        if not self.examples:
            return []
        docs = [_features(x["input"]) for x in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (f + 1)) for t, f in df.items()}

        def vector(counts: Counter[str]) -> dict[str, float]:
            raw = {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        q = vector(_features(query))
        scored = []
        for item, doc in zip(self.examples, docs):
            d = vector(doc)
            score = sum(v * d.get(t, 0) for t, v in q.items())
            if score > 0:
                scored.append((score, item))
        return sorted(scored, key=lambda pair: pair[0], reverse=True)[:self.neighbor_count]

    def _format(self, query: str) -> tuple[str, str]:
        neighbors = self._neighbors(query)
        totals: defaultdict[str, float] = defaultdict(float)
        for score, item in neighbors:
            totals[item["target"]] += score * score
        ranked_labels = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        total = sum(totals.values()) or 1.0
        support = "\n".join(f"- {label}: {weight / total:.0%} of weighted support" for label, weight in ranked_labels[:8]) or "No similar labeled examples."
        parts, used = [], 0
        for score, item in neighbors[:self.example_count]:
            part = f"Problem: {item['input']}\nVerified answer: {item['target']}"
            remaining = self.context_budget - used
            if remaining < 120:
                break
            if len(part) > remaining:
                part = part[:remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return support, self._last_context or "No similar case examples have been learned yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        support, examples = self._format(input)
        response = self.call_llm(_PROMPT.format(support=support, examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._last_context.split("\n\n")) if self._last_context else 0}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["input"], x["target"]) for x in self.examples}
        for row in batch_results:
            item = {"input": _source(row), "target": str(row["ground_truth"])}
            key = (item["input"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
