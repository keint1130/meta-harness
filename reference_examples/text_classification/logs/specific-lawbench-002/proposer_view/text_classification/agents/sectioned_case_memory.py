"""Retrieve compact examples with their fact narrative and conclusion kept distinct."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_BOILERPLATE = re.compile(r"请你模拟|请先进行推理|格式示例|案件事实|Return JSON|Current problem", re.I)
_PROMPT = """The examples separate the underlying events from procedural and evidentiary detail. Compare the event facts first, then use the case conclusion only as an analogy. The examples are not rules; follow the current problem and give a complete answer in its requested form.

{examples}

Current problem:
{query}

Return JSON with a concise final_answer.
{{"reasoning":"brief comparison","final_answer":"answer"}}"""


def _source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("raw_question") or metadata.get("raw_question") or metadata.get("question") or row.get("input", ""))


def _sections(text: str) -> tuple[str, str]:
    text = _BOILERPLATE.sub(" ", text)
    text = " ".join(text.split())
    # Keep an early event summary and a later charging/conclusion passage as separate fields.
    conclusion_marks = ("其行为已构成", "构成××", "应当以××", "行为触犯", "公诉机关认为", "检察机关认为")
    positions = [text.find(mark) for mark in conclusion_marks if text.find(mark) >= 0]
    conclusion = text[max(positions):] if positions else text[-650:]
    facts = text[: max(650, min(len(text), 1450))]
    if len(facts) > 1450:
        facts = facts[:1050] + " … " + facts[-390:]
    conclusion = conclusion[-650:]
    return facts.strip(), conclusion.strip()


def _features(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.update(run[i:i + 2] for i in range(len(run) - 1))
            result.update(run[i:i + 3] for i in range(len(run) - 2))
        else:
            result[run] += 1
    return result


class Candidate(MemorySystem):
    """Index event summaries and conclusion passages independently."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.budget = 10000
        self._last = ""

    def _rank(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        qfacts, qconclusion = _sections(query)
        docs = [_features(x["facts"] + " " + x["conclusion"]) for x in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (f + 1)) for t, f in df.items()}

        def vector(counts: Counter[str]) -> dict[str, float]:
            raw = {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in counts.items() if c}
            norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        qf, qc = vector(_features(qfacts)), vector(_features(qconclusion))
        ranked = []
        for i, item in enumerate(self.examples):
            dfv = vector(_features(item["facts"]))
            dcv = vector(_features(item["conclusion"]))
            fact_score = sum(v * dfv.get(t, 0) for t, v in qf.items())
            conclusion_score = sum(v * dcv.get(t, 0) for t, v in qc.items())
            ranked.append((0.78 * fact_score + 0.22 * conclusion_score, i))
        return sorted(ranked, reverse=True)

    def _format(self, query: str) -> str:
        parts, used = [], 0
        for score, i in self._rank(query)[:12]:
            if score <= 0:
                continue
            x = self.examples[i]
            block = f"Event facts: {x['facts']}\nCase conclusion: {x['conclusion']}\nVerified answer: {x['target']}"
            if used + len(block) > self.budget:
                continue
            parts.append(block)
            used += len(block) + 2
        self._last = "\n\n".join(parts)
        return self._last or "No relevant labeled cases have been learned yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format(input)
        response = self.call_llm(_PROMPT.format(examples=examples, query=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._last.split("\n\n")) if self._last else 0}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["facts"], x["conclusion"], x["target"]) for x in self.examples}
        for row in batch_results:
            facts, conclusion = _sections(_source(row))
            target = str(row["ground_truth"])
            key = (facts, conclusion, target)
            if facts and key not in seen:
                self.examples.append({"facts": facts, "conclusion": conclusion, "target": target})
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last = ""
