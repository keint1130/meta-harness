"""Retrieve examples by factual overlap and joint answer-set structure."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Use the verified examples as analogies. The answer bundles show which outputs can occur together; compare the current facts with the examples and include every distinct answer supported by the current problem. Do not add an answer only because it appears in a bundle. Preserve the requested output form.

{examples}

Current problem:
{query}

Return JSON with final_answer.
{{"reasoning":"brief evidence check for each answer component","final_answer":"complete answer"}}"""


def _source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("raw_question") or metadata.get("raw_question") or metadata.get("question") or row.get("input", ""))


def _tokens(text: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            out.update(run[i:i + 2] for i in range(len(run) - 1))
            out.update(run[i:i + 3] for i in range(len(run) - 2))
        else:
            out[run] += 1
    return out


def _bundle(target: str) -> tuple[str, ...]:
    # Keep the answer components exactly as learned, normalizing only separators and whitespace.
    return tuple(sorted({part.strip() for part in re.split(r"[;；、]", target) if part.strip()}))


class Candidate(MemorySystem):
    """Keep label co-occurrence as a first-class retrieval feature."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.budget = 11500
        self.max_examples = 16
        self._last = ""

    def _rank(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        docs = [_tokens(x["input"]) for x in self.examples]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (f + 1)) for t, f in df.items()}
        q = _tokens(query)

        def vec(c: Counter[str]) -> dict[str, float]:
            raw = {t: (1 + math.log(v)) * idf.get(t, 0) for t, v in c.items() if v}
            norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        qv = vec(q)
        return sorted(((sum(v * vec(d).get(t, 0) for t, v in qv.items()), i) for i, d in enumerate(docs)), reverse=True)

    def _select(self, query: str) -> list[int]:
        ranked = [(score, i) for score, i in self._rank(query) if score > 0]
        selected: list[int] = []
        bundle_counts: Counter[tuple[str, ...]] = Counter()
        component_counts: Counter[str] = Counter()
        while ranked and len(selected) < self.max_examples:
            best_pos, best_value = -1, -1.0
            for pos, (similarity, i) in enumerate(ranked):
                bundle = self.examples[i]["bundle"]
                novelty = 1.0 / (1 + bundle_counts[bundle])
                rare_component = sum(1 / (1 + component_counts[label]) for label in bundle)
                value = 0.74 * similarity + 0.18 * novelty + 0.08 * min(1.0, rare_component)
                if value > best_value:
                    best_pos, best_value = pos, value
            if best_pos < 0:
                break
            _, i = ranked.pop(best_pos)
            selected.append(i)
            bundle_counts[self.examples[i]["bundle"]] += 1
            component_counts.update(self.examples[i]["bundle"])
        return selected

    def _format(self, query: str) -> str:
        parts, used = [], 0
        for i in self._select(query):
            ex = self.examples[i]
            text = ex["input"]
            if len(text) > 1450:
                text = text[:1030] + " … " + text[-417:]
            bundle = " ; ".join(ex["bundle"]) or ex["target"]
            block = f"Related problem: {text}\nVerified answer bundle: {bundle}"
            if used + len(block) > self.budget:
                continue
            parts.append(block)
            used += len(block) + 2
        self._last = "\n\n".join(parts)
        return self._last or "No related answer bundles are available yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format(input)
        response = self.call_llm(_PROMPT.format(examples=examples, query=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._last.split("\n\n")) if self._last else 0}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["input"], x["target"]) for x in self.examples}
        for row in batch_results:
            text, target = _source(row), str(row["ground_truth"])
            if (text, target) not in seen:
                self.examples.append({"input": text, "target": target, "bundle": _bundle(target)})
                seen.add((text, target))

    def get_context_length(self) -> int:
        return len(self._last)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last = ""
