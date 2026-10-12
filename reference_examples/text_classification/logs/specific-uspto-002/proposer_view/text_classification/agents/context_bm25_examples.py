"""Retrieve a small set of context-matched cases with IDF-weighted lexical ranking."""

import json
import math
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use the verified examples as limited precedent. Transfer a pattern
only when it is supported by the current task; do not copy an answer merely
because it is the nearest match. Follow the original task's requested format.

{examples}

Current task:
{input}

Return a JSON object with "reasoning" and "final_answer" fields.
"""


def _case(text: str) -> str:
    match = re.search(
        r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:\s*(.*?)(?:\bAnswer\s*:|$)",
        text,
    )
    if match:
        return "Context: " + match.group(1).strip() + " Input: " + match.group(2).strip()
    return text.strip()


def _context(text: str) -> str:
    match = re.search(r"(?is)\bContext\s*:\s*(.*?)\bInput\s*:", text)
    return re.sub(r"\s+", " ", match.group(1).strip().casefold()) if match else ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.casefold())


class Candidate(MemorySystem):
    """Use corpus-aware term weighting instead of raw set overlap."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.max_examples = 2

    def _rank(self, input: str) -> list[tuple[float, dict[str, str]]]:
        query = _case(input)
        route = _context(query)
        query_terms = _tokens(query)
        docs = [ex for ex in self.examples if _context(ex["case"]) == route]
        if not docs:
            return []
        doc_terms = [_tokens(ex["case"]) for ex in docs]
        df: dict[str, int] = {}
        for terms in doc_terms:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        avg_len = sum(map(len, doc_terms)) / len(doc_terms) or 1.0
        ranked = []
        for ex, terms in zip(docs, doc_terms):
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(docs) - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                score += idf * frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * len(terms) / avg_len)
                )
            ranked.append((score, ex))
        ranked.sort(key=lambda item: (item[0], item[1]["case"]), reverse=True)
        return ranked[: self.max_examples]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        selected = self._rank(input)
        examples = "\n\n".join(
            f"Verified example {i}:\n{ex['case']}\nVerified answer: {ex['target']}"
            for i, (_, ex) in enumerate(selected, 1)
        ) or "No prior examples are available."
        response = self.call_llm(PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "retrieved_examples": len(selected),
            "retrieval_scores": [score for score, _ in selected],
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        known = {(ex["case"], ex["target"]) for ex in self.examples}
        for row in batch_results:
            case = _case(str(row.get("input", "")))
            target = str(row.get("ground_truth", "")).strip()
            key = (case, target)
            if case and target and key not in known:
                self.examples.append({"case": case, "target": target})
                known.add(key)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        lengths = []
        for ex in self.examples:
            selected = self._rank(ex["case"])
            lengths.append(
                sum(len(item["case"]) + len(item["target"]) + 48 for _, item in selected)
            )
        return round(sum(lengths) / len(lengths))

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
