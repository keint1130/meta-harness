"""Collapse repeated labeled cases before query-time retrieval."""

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Use the verified case references as demonstrations. Repeated or similar text can occur, so focus on distinct facts and use a case only when its operative facts support the same conclusion. Preserve the answer form requested in the current problem.

{examples}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _tokens(text: str) -> Counter[str]:
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


def _key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class Candidate(MemorySystem):
    """Store one representative per exact normalized case and label."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.context_budget = 10000
        self.max_examples = 10
        self._last_context = ""

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        if not self.examples:
            return []
        docs = [_tokens(x["input"]) for x in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (f + 1)) for t, f in df.items()}
        q = _tokens(query)
        qweight = {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in q.items()}
        qnorm = math.sqrt(sum(v * v for v in qweight.values())) or 1.0
        scored = []
        for i, doc in enumerate(docs):
            weight = {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in doc.items()}
            norm = math.sqrt(sum(v * v for v in weight.values())) or 1.0
            score = sum(v * weight.get(t, 0) for t, v in qweight.items()) / (qnorm * norm)
            scored.append((score, i))
        return [self.examples[i] for score, i in sorted(scored, reverse=True)[:self.max_examples] if score > 0]

    def _format(self, query: str) -> str:
        parts, used = [], 0
        for ex in self._retrieve(query):
            part = f"Case: {ex['input']}\nVerified answer: {ex['target']}"
            remaining = self.context_budget - used
            if remaining < 120:
                break
            if len(part) > remaining:
                part = part[:remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context = "\n\n".join(parts)
        return self._last_context or "No distinct case references have been learned yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._last_context.split("\n\n")) if self._last_context else 0, "unique_cases": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        indexes = {(_key(x["input"]), x["target"]): x for x in self.examples}
        for row in batch_results:
            text, target = _source(row), str(row["ground_truth"])
            key = (_key(text), target)
            if key in indexes:
                indexes[key]["count"] += 1
            else:
                item = {"input": text, "target": target, "count": 1}
                self.examples.append(item)
                indexes[key] = item

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
