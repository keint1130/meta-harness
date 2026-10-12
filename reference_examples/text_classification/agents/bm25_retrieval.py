"""Query-conditioned few-shot retrieval with a fixed context budget."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """Use the related examples as demonstrations. Their answers are labeled feedback.

{examples}

Problem:
{input}

Follow the problem's own instructions and return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]"}}"""


def _features(text: str) -> list[str]:
    features: list[str] = []
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            features.extend(run[i : i + 2] for i in range(len(run) - 1))
            features.extend(run[i : i + 3] for i in range(len(run) - 2))
        else:
            features.append(run)
    return features


def _source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        row.get("raw_question")
        or metadata.get("raw_question")
        or metadata.get("question")
        or row.get("input", "")
    )


def _clip(text: str, limit: int = 1500) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + " … " + text[-(limit - head - 3) :]


class Candidate(MemorySystem):
    """Select demonstrations by query similarity rather than recency or order."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 14000
        self._last_context_len = 0

    def _retrieve(self, query: str, limit: int = 14) -> list[dict[str, str]]:
        if not self.examples:
            return []
        docs = [_features(ex["input"]) for ex in self.examples]
        doc_freq: Counter[str] = Counter()
        for doc in docs:
            doc_freq.update(set(doc))
        avg_len = sum(map(len, docs)) / max(1, len(docs))
        qtf = Counter(_features(query))
        n_docs = len(docs)
        ranked: list[tuple[float, int]] = []
        for index, doc in enumerate(docs):
            tf = Counter(doc)
            score = 0.0
            for term, q_count in qtf.items():
                term_count = tf.get(term, 0)
                if not term_count:
                    continue
                df = doc_freq[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = term_count + 1.2 * (
                    0.25 + 0.75 * len(doc) / max(1.0, avg_len)
                )
                score += min(q_count, 2) * idf * term_count * 2.2 / denom
            ranked.append((score, index))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [self.examples[index] for _, index in ranked[:limit]]

    def _format_examples(self, query: str) -> str:
        selected = self._retrieve(query)
        parts: list[str] = []
        used = 0
        for example in selected:
            part = f"Q: {_clip(example['input'])}\nA: {example['target']}"
            remaining = self.context_budget - used
            if remaining <= 0:
                break
            if len(part) > remaining:
                if remaining < 80:
                    break
                part = part[: remaining - 3] + "..."
            parts.append(part)
            used += len(part) + 2
        self._last_context_len = used
        return "\n\n".join(parts) or "No learned examples yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples(input)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "retrieved_examples": 0 if examples == "No learned examples yet." else len(self._retrieve(input)),
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for row in batch_results:
            self.examples.append(
                {"input": _source_text(row), "target": str(row["ground_truth"])}
            )

    def get_context_length(self) -> int:
        return self._last_context_len

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])

